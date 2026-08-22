-- Fix helper reference after moving RLS helper functions into app_private.

begin;

create or replace function app_private.is_campaign_dm(target_campaign_id uuid)
returns boolean
language sql
stable
security definer
set search_path = public, app_private
as $$
  select app_private.is_platform_admin() or exists(
    select 1
    from public.campaign_members cm
    where cm.campaign_id = target_campaign_id
      and cm.user_id = auth.uid()
      and cm.role = 'dm'
      and cm.status = 'active'
  );
$$;

grant execute on function app_private.is_campaign_dm(uuid) to authenticated;

commit;
