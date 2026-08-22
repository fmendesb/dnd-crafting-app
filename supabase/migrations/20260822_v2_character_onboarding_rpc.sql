-- Transactional player character onboarding.
begin;

create or replace function public.create_my_character(
  target_campaign_id uuid,
  character_name text,
  profession_one uuid default null,
  profession_two uuid default null
)
returns uuid
language plpgsql
security definer
set search_path = public, app_private, auth
as $$
declare
  uid uuid := auth.uid();
  membership_id uuid;
  new_character_id uuid;
  cg_enabled boolean;
begin
  if uid is null then raise exception 'You must be signed in.'; end if;
  if char_length(trim(coalesce(character_name,''))) < 1 then raise exception 'Character name cannot be empty.'; end if;

  select cm.id into membership_id
  from public.campaign_members cm
  join public.campaigns c on c.id=cm.campaign_id
  where cm.campaign_id=target_campaign_id and cm.user_id=uid and cm.role='player' and cm.status='active' and c.status='active';
  if membership_id is null then raise exception 'You are not an active player in this campaign.'; end if;

  if exists(select 1 from public.characters ch where ch.owner_membership_id=membership_id and ch.status='active') then
    raise exception 'You already have an active character in this campaign.';
  end if;

  select exists(
    select 1 from public.campaign_modules cm
    join public.app_modules am on am.id=cm.module_id
    where cm.campaign_id=target_campaign_id and cm.enabled and am.code='crafting_gathering'
  ) into cg_enabled;

  if cg_enabled then
    if profession_one is null or profession_two is null or profession_one=profession_two then raise exception 'Choose two different professions.'; end if;
    if (select count(*) from public.professions p where p.id in (profession_one, profession_two) and p.is_active) <> 2 then raise exception 'One or more selected professions are invalid.'; end if;
  end if;

  insert into public.characters(campaign_id,owner_membership_id,name)
  values(target_campaign_id,membership_id,trim(character_name))
  returning id into new_character_id;

  if cg_enabled then
    insert into public.character_professions(character_id,slot,profession_id) values
      (new_character_id,1,profession_one),
      (new_character_id,2,profession_two);
  end if;

  return new_character_id;
end;
$$;
revoke all on function public.create_my_character(uuid,text,uuid,uuid) from public, anon;
grant execute on function public.create_my_character(uuid,text,uuid,uuid) to authenticated;

commit;
