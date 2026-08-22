-- Enforce module entitlement on both inserts and re-enabling updates.

begin;

create or replace function app_private.enforce_module_entitlement()
returns trigger
language plpgsql
security definer
set search_path = public, app_private
as $$
declare
  enabled_count integer;
  limit_count integer;
begin
  if not new.enabled then
    return new;
  end if;

  if app_private.is_platform_admin() then
    return new;
  end if;

  if not app_private.is_campaign_dm(new.campaign_id) then
    raise exception 'Only the campaign DM can enable modules';
  end if;

  if tg_op = 'UPDATE' and old.enabled = true then
    return new;
  end if;

  select count(*) into enabled_count
  from public.campaign_modules cm
  where cm.campaign_id = new.campaign_id
    and cm.enabled = true
    and (tg_op <> 'UPDATE' or cm.module_id <> new.module_id);

  limit_count := app_private.max_modules_for(auth.uid());

  if enabled_count >= limit_count then
    raise exception 'Your plan allows % enabled module(s) per owned campaign', limit_count;
  end if;

  return new;
end;
$$;

revoke execute on function app_private.enforce_module_entitlement() from public, anon, authenticated;

drop trigger if exists campaign_modules_entitlement_guard on public.campaign_modules;
create trigger campaign_modules_entitlement_guard
before insert or update of enabled on public.campaign_modules
for each row execute function app_private.enforce_module_entitlement();

commit;
