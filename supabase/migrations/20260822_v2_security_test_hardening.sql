begin;

-- Least-privilege table grants. RLS remains the primary row-level boundary,
-- but clients should not receive blanket table capabilities they never need.
revoke all privileges on table
  public.legacy_v1_player_state,
  public.legacy_v1_keepalive,
  public.platform_admins
from anon, authenticated;

revoke all privileges on table
  public.profiles,
  public.plans,
  public.subscriptions,
  public.entitlement_overrides,
  public.campaigns,
  public.campaign_members,
  public.app_modules,
  public.campaign_modules,
  public.content_packs,
  public.campaign_content_packs,
  public.campaign_invitations,
  public.characters,
  public.professions,
  public.character_professions
from anon;

revoke all privileges on table
  public.profiles,
  public.plans,
  public.subscriptions,
  public.entitlement_overrides,
  public.campaigns,
  public.campaign_members,
  public.app_modules,
  public.campaign_modules,
  public.content_packs,
  public.campaign_content_packs,
  public.campaign_invitations,
  public.characters,
  public.professions,
  public.character_professions
from authenticated;

grant select, insert, update on public.profiles to authenticated;
grant select on public.plans, public.subscriptions, public.entitlement_overrides,
  public.app_modules, public.content_packs, public.professions to authenticated;
grant select, insert, update, delete on public.campaigns, public.campaign_members,
  public.campaign_modules, public.campaign_content_packs, public.campaign_invitations,
  public.characters, public.character_professions to authenticated;

-- Make policy role intent explicit.
alter policy app_modules_read_authenticated on public.app_modules to authenticated;
alter policy campaign_content_packs_manage_dm on public.campaign_content_packs to authenticated;
alter policy campaign_content_packs_read_member on public.campaign_content_packs to authenticated;
alter policy campaign_invitations_manage_dm on public.campaign_invitations to authenticated;
alter policy campaign_members_delete_dm on public.campaign_members to authenticated;
alter policy campaign_members_insert_dm on public.campaign_members to authenticated;
alter policy campaign_members_read_member on public.campaign_members to authenticated;
alter policy campaign_members_update_dm on public.campaign_members to authenticated;
alter policy campaign_modules_delete_dm on public.campaign_modules to authenticated;
alter policy campaign_modules_insert_entitled_dm on public.campaign_modules to authenticated;
alter policy campaign_modules_read_member on public.campaign_modules to authenticated;
alter policy campaign_modules_update_dm on public.campaign_modules to authenticated;
alter policy campaigns_delete_dm on public.campaigns to authenticated;
alter policy campaigns_insert_entitled_owner on public.campaigns to authenticated;
alter policy campaigns_read_member on public.campaigns to authenticated;
alter policy campaigns_update_dm on public.campaigns to authenticated;
alter policy characters_delete_owner_or_dm on public.characters to authenticated;
alter policy characters_read_member on public.characters to authenticated;
alter policy content_packs_read_authenticated on public.content_packs to authenticated;
alter policy entitlement_overrides_read_self on public.entitlement_overrides to authenticated;
alter policy plans_read_authenticated on public.plans to authenticated;
alter policy profiles_insert_self on public.profiles to authenticated;
alter policy profiles_select_self_or_admin on public.profiles to authenticated;
alter policy profiles_update_self on public.profiles to authenticated;
alter policy subscriptions_read_self on public.subscriptions to authenticated;

-- Fix character insert authorization: the selected membership must belong to the
-- same campaign as the new character. The previous tautology could allow a player
-- to inject a character row into another campaign.
drop policy if exists characters_insert_self_or_dm on public.characters;
create policy characters_insert_self_or_dm
on public.characters
for insert
to authenticated
with check (
  app_private.is_platform_admin()
  or app_private.is_campaign_dm(campaign_id)
  or (
    owner_membership_id is not null
    and exists (
      select 1
      from public.campaign_members cm
      join public.campaigns c on c.id = cm.campaign_id
      where cm.id = characters.owner_membership_id
        and cm.campaign_id = characters.campaign_id
        and cm.user_id = auth.uid()
        and cm.status = 'active'
        and cm.role = 'player'
        and c.status = 'active'
    )
  )
);

-- Keep player-owned characters anchored to their original campaign/membership.
create or replace function app_private.protect_character_identity_fields()
returns trigger
language plpgsql
set search_path = public, app_private, auth
as $$
begin
  if auth.uid() is null then
    return new;
  end if;

  if app_private.is_platform_admin() or app_private.is_campaign_dm(old.campaign_id) then
    return new;
  end if;

  if new.campaign_id is distinct from old.campaign_id
     or new.owner_membership_id is distinct from old.owner_membership_id then
    raise exception 'Players cannot move or reassign characters between campaigns.';
  end if;

  return new;
end;
$$;

revoke all on function app_private.protect_character_identity_fields() from public, anon, authenticated;

drop trigger if exists protect_character_identity_fields on public.characters;
create trigger protect_character_identity_fields
before update on public.characters
for each row execute function app_private.protect_character_identity_fields();

-- Harden invitation acceptance against self-demoting a campaign DM and make
-- already-active player membership idempotent.
create or replace function public.accept_campaign_invitation(raw_token text)
returns uuid
language plpgsql
security definer
set search_path = pg_catalog, public, app_private, auth, extensions
as $$
declare
  inv public.campaign_invitations%rowtype;
  uid uuid := auth.uid();
  user_email text := lower(coalesce(auth.jwt()->>'email',''));
  existing_role text;
  existing_status text;
begin
  if uid is null then raise exception 'You must be signed in to accept an invitation.'; end if;
  if raw_token is null or length(trim(raw_token)) < 16 then raise exception 'Invitation token is invalid.'; end if;

  select * into inv
  from public.campaign_invitations
  where token_hash = encode(extensions.digest(raw_token,'sha256'),'hex')
  for update;

  if not found then raise exception 'Invitation is invalid.'; end if;
  if inv.revoked_at is not null then raise exception 'Invitation has been revoked.'; end if;
  if inv.expires_at is not null and inv.expires_at <= now() then raise exception 'Invitation has expired.'; end if;
  if inv.max_uses is not null and inv.use_count >= inv.max_uses then raise exception 'Invitation has already been used.'; end if;
  if inv.kind='email' and lower(inv.email) <> user_email then raise exception 'This invitation was sent to a different email address.'; end if;
  if not exists(select 1 from public.campaigns c where c.id=inv.campaign_id and c.status='active') then raise exception 'Campaign is not active.'; end if;

  select cm.role, cm.status into existing_role, existing_status
  from public.campaign_members cm
  where cm.campaign_id=inv.campaign_id and cm.user_id=uid;

  if existing_role = 'dm' then
    raise exception 'You are already the Dungeon Master of this campaign.';
  end if;

  if existing_role = 'player' and existing_status = 'active' then
    return inv.campaign_id;
  end if;

  if existing_role is null then
    if not app_private.can_participate_in_another_campaign(uid) then
      raise exception 'Your plan does not allow another campaign.';
    end if;
    insert into public.campaign_members(campaign_id,user_id,role,status)
    values(inv.campaign_id,uid,'player','active');
  else
    update public.campaign_members
    set role='player', status='active', left_at=null
    where campaign_id=inv.campaign_id and user_id=uid;
  end if;

  update public.campaign_invitations set use_count=use_count+1 where id=inv.id;
  return inv.campaign_id;
end;
$$;

revoke all on function public.accept_campaign_invitation(text) from public, anon;
grant execute on function public.accept_campaign_invitation(text) to authenticated;

-- Tighten character RPC search path and function privileges.
alter function public.create_my_character(uuid,text,uuid,uuid)
  set search_path = pg_catalog, public, app_private, auth;
revoke all on function public.create_my_character(uuid,text,uuid,uuid) from public, anon;
grant execute on function public.create_my_character(uuid,text,uuid,uuid) to authenticated;

commit;
