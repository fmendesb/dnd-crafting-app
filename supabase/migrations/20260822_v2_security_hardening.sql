-- Security hardening applied after the V2 foundation migration.

begin;

alter table if exists public.legacy_v1_player_state enable row level security;
alter table if exists public.legacy_v1_keepalive enable row level security;

do $$
begin
  if exists (
    select 1 from pg_proc p join pg_namespace n on n.oid=p.pronamespace
    where n.nspname='public' and p.proname='set_player_state_updated_at'
  ) then
    execute 'alter function public.set_player_state_updated_at() set search_path = public';
  end if;
end $$;

create schema if not exists app_private;
revoke all on schema app_private from public;
grant usage on schema app_private to authenticated;

do $$
begin
  if to_regprocedure('public.is_platform_admin()') is not null then
    alter function public.is_platform_admin() set schema app_private;
  end if;
  if to_regprocedure('public.is_campaign_member(uuid)') is not null then
    alter function public.is_campaign_member(uuid) set schema app_private;
  end if;
  if to_regprocedure('public.is_campaign_dm(uuid)') is not null then
    alter function public.is_campaign_dm(uuid) set schema app_private;
  end if;
  if to_regprocedure('public.owns_character(uuid)') is not null then
    alter function public.owns_character(uuid) set schema app_private;
  end if;
end $$;

grant execute on function app_private.is_platform_admin() to authenticated;
grant execute on function app_private.is_campaign_member(uuid) to authenticated;
grant execute on function app_private.is_campaign_dm(uuid) to authenticated;
grant execute on function app_private.owns_character(uuid) to authenticated;

commit;
