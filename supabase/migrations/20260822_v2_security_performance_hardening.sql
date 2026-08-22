begin;

-- Cover foreign keys used by authorization and common joins.
create index if not exists campaign_content_packs_content_pack_idx on public.campaign_content_packs(content_pack_id);
create index if not exists campaign_invitations_created_by_idx on public.campaign_invitations(created_by);
create index if not exists campaign_modules_module_id_idx on public.campaign_modules(module_id);
create index if not exists entitlement_overrides_created_by_idx on public.entitlement_overrides(created_by);
create index if not exists entitlement_overrides_plan_id_idx on public.entitlement_overrides(plan_id);
create index if not exists subscriptions_plan_id_idx on public.subscriptions(plan_id);

-- Avoid per-row re-evaluation of auth.uid() in RLS policies.
alter policy app_modules_read_authenticated on public.app_modules
using (((select auth.uid()) is not null) and is_active);

alter policy content_packs_read_authenticated on public.content_packs
using (((select auth.uid()) is not null) and is_active);

alter policy plans_read_authenticated on public.plans
using (((select auth.uid()) is not null) and is_active);

alter policy entitlement_overrides_read_self on public.entitlement_overrides
using ((user_id = (select auth.uid())) or (select app_private.is_platform_admin()));

alter policy subscriptions_read_self on public.subscriptions
using ((user_id = (select auth.uid())) or (select app_private.is_platform_admin()));

alter policy profiles_insert_self on public.profiles
with check ((user_id = (select auth.uid())) or (select app_private.is_platform_admin()));

alter policy profiles_select_self_or_admin on public.profiles
using ((user_id = (select auth.uid())) or (select app_private.is_platform_admin()));

alter policy profiles_update_self on public.profiles
using ((user_id = (select auth.uid())) or (select app_private.is_platform_admin()))
with check ((user_id = (select auth.uid())) or (select app_private.is_platform_admin()));

alter policy campaigns_delete_dm on public.campaigns
using ((select app_private.is_platform_admin()) or owner_user_id = (select auth.uid()));

alter policy campaigns_insert_entitled_owner on public.campaigns
with check (((owner_user_id = (select auth.uid())) and app_private.can_participate_in_another_campaign((select auth.uid()))) or (select app_private.is_platform_admin()));

alter policy campaigns_read_member on public.campaigns
using ((select app_private.is_platform_admin()) or app_private.is_campaign_member(id) or owner_user_id = (select auth.uid()));

alter policy campaigns_update_dm on public.campaigns
using (app_private.is_campaign_dm(id) or owner_user_id = (select auth.uid()))
with check (app_private.is_campaign_dm(id) or owner_user_id = (select auth.uid()));

alter policy campaign_members_insert_dm on public.campaign_members
with check (
  (select app_private.is_platform_admin())
  or app_private.is_campaign_dm(campaign_id)
  or (
    role='dm'
    and user_id=(select auth.uid())
    and exists(select 1 from public.campaigns c where c.id=campaign_members.campaign_id and c.owner_user_id=(select auth.uid()))
  )
);

alter policy campaign_members_read_member on public.campaign_members
using ((select app_private.is_platform_admin()) or app_private.is_campaign_member(campaign_id) or user_id=(select auth.uid()));

alter policy campaign_modules_insert_entitled_dm on public.campaign_modules
with check (
  (select app_private.is_platform_admin())
  or (
    app_private.is_campaign_dm(campaign_id)
    and (select count(*) from public.campaign_modules cm where cm.campaign_id=campaign_modules.campaign_id and cm.enabled)
      < app_private.max_modules_for((select auth.uid()))
  )
);

alter policy characters_insert_self_or_dm on public.characters
with check (
  (select app_private.is_platform_admin())
  or app_private.is_campaign_dm(campaign_id)
  or (
    owner_membership_id is not null
    and exists (
      select 1
      from public.campaign_members cm
      join public.campaigns c on c.id=cm.campaign_id
      where cm.id=characters.owner_membership_id
        and cm.campaign_id=characters.campaign_id
        and cm.user_id=(select auth.uid())
        and cm.status='active'
        and cm.role='player'
        and c.status='active'
    )
  )
);

-- Avoid duplicate SELECT policy evaluation for campaign content packs.
drop policy if exists campaign_content_packs_manage_dm on public.campaign_content_packs;
create policy campaign_content_packs_insert_dm on public.campaign_content_packs
for insert to authenticated
with check (app_private.is_campaign_dm(campaign_id) or (select app_private.is_platform_admin()));
create policy campaign_content_packs_update_dm on public.campaign_content_packs
for update to authenticated
using (app_private.is_campaign_dm(campaign_id) or (select app_private.is_platform_admin()))
with check (app_private.is_campaign_dm(campaign_id) or (select app_private.is_platform_admin()));
create policy campaign_content_packs_delete_dm on public.campaign_content_packs
for delete to authenticated
using (app_private.is_campaign_dm(campaign_id) or (select app_private.is_platform_admin()));

commit;
