-- V2 invitations + profession selection foundation.
begin;

create table if not exists public.professions (
  id uuid primary key default gen_random_uuid(),
  code text not null unique,
  name text not null unique,
  category text not null check (category in ('gathering','crafting')),
  is_active boolean not null default true,
  created_at timestamptz not null default now()
);

insert into public.professions(code,name,category) values
('mining','Mining','gathering'),
('hunting','Hunting','gathering'),
('herbalism','Herbalism','gathering'),
('arcana_extraction','Arcana Extraction','gathering'),
('blacksmithing','Blacksmithing','crafting'),
('woodcrafter','Woodcrafter','crafting'),
('cooking','Cooking','crafting'),
('tailoring','Tailoring','crafting'),
('scribing','Scribing','crafting'),
('enchanting','Enchanting','crafting')
on conflict (code) do update set name=excluded.name, category=excluded.category, is_active=true;

create table if not exists public.character_professions (
  character_id uuid not null references public.characters(id) on delete cascade,
  slot smallint not null check (slot in (1,2)),
  profession_id uuid not null references public.professions(id),
  level integer not null default 1 check (level >= 1),
  xp numeric not null default 0 check (xp >= 0),
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  primary key(character_id,slot),
  unique(character_id,profession_id)
);
create index if not exists character_professions_profession_idx on public.character_professions(profession_id);
create trigger character_professions_set_updated_at before update on public.character_professions for each row execute function public.set_updated_at();

alter table public.professions enable row level security;
alter table public.character_professions enable row level security;

create policy professions_read_authenticated on public.professions for select to authenticated using (is_active);
create policy character_professions_read_member on public.character_professions for select to authenticated
using (exists(select 1 from public.characters ch where ch.id=character_id and app_private.is_campaign_member(ch.campaign_id)) or app_private.is_platform_admin());
create policy character_professions_insert_owner_or_dm on public.character_professions for insert to authenticated
with check (exists(select 1 from public.characters ch where ch.id=character_id and (app_private.owns_character(ch.id) or app_private.is_campaign_dm(ch.campaign_id))) or app_private.is_platform_admin());
create policy character_professions_update_owner_or_dm on public.character_professions for update to authenticated
using (exists(select 1 from public.characters ch where ch.id=character_id and (app_private.owns_character(ch.id) or app_private.is_campaign_dm(ch.campaign_id))) or app_private.is_platform_admin())
with check (exists(select 1 from public.characters ch where ch.id=character_id and (app_private.owns_character(ch.id) or app_private.is_campaign_dm(ch.campaign_id))) or app_private.is_platform_admin());
create policy character_professions_delete_owner_or_dm on public.character_professions for delete to authenticated
using (exists(select 1 from public.characters ch where ch.id=character_id and (app_private.owns_character(ch.id) or app_private.is_campaign_dm(ch.campaign_id))) or app_private.is_platform_admin());

create or replace function public.accept_campaign_invitation(raw_token text)
returns uuid
language plpgsql
security definer
set search_path = public, app_private, auth, extensions
as $$
declare
  inv public.campaign_invitations%rowtype;
  uid uuid := auth.uid();
  user_email text := lower(coalesce(auth.jwt()->>'email',''));
  membership_id uuid;
begin
  if uid is null then raise exception 'You must be signed in to accept an invitation.'; end if;
  if raw_token is null or length(trim(raw_token)) < 16 then raise exception 'Invitation token is invalid.'; end if;

  select * into inv from public.campaign_invitations
  where token_hash = encode(digest(raw_token,'sha256'),'hex') for update;
  if not found then raise exception 'Invitation is invalid.'; end if;
  if inv.revoked_at is not null then raise exception 'Invitation has been revoked.'; end if;
  if inv.expires_at is not null and inv.expires_at <= now() then raise exception 'Invitation has expired.'; end if;
  if inv.max_uses is not null and inv.use_count >= inv.max_uses then raise exception 'Invitation has already been used.'; end if;
  if inv.kind='email' and lower(inv.email) <> user_email then raise exception 'This invitation was sent to a different email address.'; end if;
  if not exists(select 1 from public.campaigns c where c.id=inv.campaign_id and c.status='active') then raise exception 'Campaign is not active.'; end if;

  select id into membership_id from public.campaign_members where campaign_id=inv.campaign_id and user_id=uid;
  if membership_id is null then
    if not app_private.can_participate_in_another_campaign(uid) then raise exception 'Your plan does not allow another campaign.'; end if;
    insert into public.campaign_members(campaign_id,user_id,role,status)
    values(inv.campaign_id,uid,'player','active') returning id into membership_id;
  else
    update public.campaign_members set role='player', status='active', left_at=null where id=membership_id;
  end if;

  update public.campaign_invitations set use_count=use_count+1 where id=inv.id;
  return inv.campaign_id;
end;
$$;
revoke all on function public.accept_campaign_invitation(text) from public, anon;
grant execute on function public.accept_campaign_invitation(text) to authenticated;

commit;
