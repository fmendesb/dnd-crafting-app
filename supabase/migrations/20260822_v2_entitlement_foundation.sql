-- V2 entitlement and campaign-bootstrap foundation.

begin;

create or replace function app_private.handle_new_user()
returns trigger
language plpgsql
security definer
set search_path = public, app_private
as $$
begin
  insert into public.profiles (user_id, display_name)
  values (new.id, coalesce(new.raw_user_meta_data->>'display_name', new.raw_user_meta_data->>'full_name'))
  on conflict (user_id) do nothing;
  return new;
end;
$$;

drop trigger if exists on_auth_user_created on auth.users;
create trigger on_auth_user_created after insert on auth.users for each row execute function app_private.handle_new_user();

create or replace function app_private.effective_plan_code(target_user_id uuid)
returns text
language sql
stable
security definer
set search_path = public, app_private
as $$
  with admin_check as (
    select exists(select 1 from public.platform_admins pa where pa.user_id = target_user_id) as is_admin
  ), override_plan as (
    select p.code
    from public.entitlement_overrides eo
    join public.plans p on p.id = eo.plan_id
    where eo.user_id = target_user_id
      and eo.starts_at <= now()
      and (eo.ends_at is null or eo.ends_at > now())
      and eo.plan_id is not null
    order by eo.created_at desc
    limit 1
  ), paid_plan as (
    select p.code
    from public.subscriptions s
    join public.plans p on p.id = s.plan_id
    where s.user_id = target_user_id
      and (s.status in ('active','trialing') or (s.status='grace' and s.grace_ends_at is not null and s.grace_ends_at > now()))
      and (s.current_period_end is null or s.current_period_end > now() or s.status='grace')
    order by s.updated_at desc
    limit 1
  )
  select case
    when (select is_admin from admin_check) then 'premium'
    when exists(select 1 from override_plan) then (select code from override_plan)
    when exists(select 1 from paid_plan) then (select code from paid_plan)
    else 'free'
  end;
$$;

create or replace function app_private.max_campaigns_for(target_user_id uuid)
returns integer
language sql
stable
security definer
set search_path = public, app_private
as $$
  select coalesce(
    (select eo.max_campaigns from public.entitlement_overrides eo where eo.user_id=target_user_id and eo.starts_at<=now() and (eo.ends_at is null or eo.ends_at>now()) and eo.max_campaigns is not null order by eo.created_at desc limit 1),
    (select p.max_campaigns from public.plans p where p.code=app_private.effective_plan_code(target_user_id)),
    1
  );
$$;

create or replace function app_private.max_modules_for(target_user_id uuid)
returns integer
language sql
stable
security definer
set search_path = public, app_private
as $$
  select coalesce(
    (select eo.max_modules_per_owned_campaign from public.entitlement_overrides eo where eo.user_id=target_user_id and eo.starts_at<=now() and (eo.ends_at is null or eo.ends_at>now()) and eo.max_modules_per_owned_campaign is not null order by eo.created_at desc limit 1),
    (select p.max_modules_per_owned_campaign from public.plans p where p.code=app_private.effective_plan_code(target_user_id)),
    1
  );
$$;

create or replace function app_private.can_participate_in_another_campaign(target_user_id uuid)
returns boolean
language sql
stable
security definer
set search_path = public, app_private
as $$
  select (select count(*) from public.campaign_members cm where cm.user_id=target_user_id and cm.status='active') < app_private.max_campaigns_for(target_user_id)
  or exists(select 1 from public.platform_admins pa where pa.user_id=target_user_id);
$$;

create or replace function app_private.after_campaign_created()
returns trigger
language plpgsql
security definer
set search_path = public, app_private
as $$
declare core_pack_id uuid;
begin
  insert into public.campaign_members(campaign_id,user_id,role,status) values(new.id,new.owner_user_id,'dm','active');
  select id into core_pack_id from public.content_packs where code='core' limit 1;
  if core_pack_id is not null then
    insert into public.campaign_content_packs(campaign_id,content_pack_id,enabled)
    values(new.id,core_pack_id,true)
    on conflict (campaign_id,content_pack_id) do update set enabled=true, updated_at=now();
  end if;
  return new;
end;
$$;

drop trigger if exists campaign_created_bootstrap on public.campaigns;
create trigger campaign_created_bootstrap after insert on public.campaigns for each row execute function app_private.after_campaign_created();

drop policy if exists campaigns_insert_owner on public.campaigns;
drop policy if exists campaigns_insert_entitled_owner on public.campaigns;
create policy campaigns_insert_entitled_owner on public.campaigns for insert
with check ((owner_user_id=auth.uid() and app_private.can_participate_in_another_campaign(auth.uid())) or app_private.is_platform_admin());

drop policy if exists campaign_modules_manage_dm on public.campaign_modules;
drop policy if exists campaign_modules_read_member on public.campaign_modules;
drop policy if exists campaign_modules_insert_entitled_dm on public.campaign_modules;
drop policy if exists campaign_modules_update_dm on public.campaign_modules;
drop policy if exists campaign_modules_delete_dm on public.campaign_modules;
create policy campaign_modules_read_member on public.campaign_modules for select using (app_private.is_platform_admin() or app_private.is_campaign_member(campaign_id));
create policy campaign_modules_insert_entitled_dm on public.campaign_modules for insert
with check (app_private.is_platform_admin() or (app_private.is_campaign_dm(campaign_id) and (select count(*) from public.campaign_modules cm where cm.campaign_id=campaign_modules.campaign_id and cm.enabled) < app_private.max_modules_for(auth.uid())));
create policy campaign_modules_update_dm on public.campaign_modules for update using (app_private.is_platform_admin() or app_private.is_campaign_dm(campaign_id)) with check (app_private.is_platform_admin() or app_private.is_campaign_dm(campaign_id));
create policy campaign_modules_delete_dm on public.campaign_modules for delete using (app_private.is_platform_admin() or app_private.is_campaign_dm(campaign_id));

grant execute on function app_private.effective_plan_code(uuid) to authenticated;
grant execute on function app_private.max_campaigns_for(uuid) to authenticated;
grant execute on function app_private.max_modules_for(uuid) to authenticated;
grant execute on function app_private.can_participate_in_another_campaign(uuid) to authenticated;

commit;
