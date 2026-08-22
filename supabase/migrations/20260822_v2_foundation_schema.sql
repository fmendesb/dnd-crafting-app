-- V2 foundation schema for the D&D Companion App sandbox.
-- Applied to Supabase project: Crafting and Gathering Test.
-- Production is intentionally untouched.

begin;

alter table if exists public.player_state rename to legacy_v1_player_state;
alter table if exists public.keepalive rename to legacy_v1_keepalive;

create extension if not exists pgcrypto;

create table if not exists public.profiles (
  user_id uuid primary key references auth.users(id) on delete cascade,
  display_name text,
  avatar_path text,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table if not exists public.platform_admins (
  user_id uuid primary key references auth.users(id) on delete cascade,
  created_at timestamptz not null default now()
);

create table if not exists public.plans (
  id uuid primary key default gen_random_uuid(),
  code text not null unique,
  name text not null,
  max_campaigns integer not null check (max_campaigns >= 0),
  max_modules_per_owned_campaign integer not null check (max_modules_per_owned_campaign >= 0),
  includes_subscription_content_packs boolean not null default false,
  is_active boolean not null default true,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

insert into public.plans (code, name, max_campaigns, max_modules_per_owned_campaign, includes_subscription_content_packs)
values ('free','Free',1,1,false), ('premium','Premium',5,3,true)
on conflict (code) do update set
  name=excluded.name,
  max_campaigns=excluded.max_campaigns,
  max_modules_per_owned_campaign=excluded.max_modules_per_owned_campaign,
  includes_subscription_content_packs=excluded.includes_subscription_content_packs,
  is_active=true,
  updated_at=now();

create table if not exists public.subscriptions (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  plan_id uuid not null references public.plans(id),
  provider text,
  provider_customer_id text,
  provider_subscription_id text,
  billing_interval text check (billing_interval in ('monthly','annual') or billing_interval is null),
  status text not null default 'inactive' check (status in ('inactive','trialing','active','past_due','grace','cancelled','expired')),
  current_period_start timestamptz,
  current_period_end timestamptz,
  grace_ends_at timestamptz,
  cancel_at_period_end boolean not null default false,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  unique (provider, provider_subscription_id)
);
create index if not exists subscriptions_user_idx on public.subscriptions(user_id);

create table if not exists public.entitlement_overrides (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  plan_id uuid references public.plans(id),
  max_campaigns integer,
  max_modules_per_owned_campaign integer,
  includes_subscription_content_packs boolean,
  reason text,
  starts_at timestamptz not null default now(),
  ends_at timestamptz,
  created_by uuid references auth.users(id),
  created_at timestamptz not null default now()
);
create index if not exists entitlement_overrides_user_idx on public.entitlement_overrides(user_id);

create table if not exists public.campaigns (
  id uuid primary key default gen_random_uuid(),
  name text not null check (char_length(trim(name)) between 1 and 120),
  owner_user_id uuid not null references auth.users(id),
  description text,
  status text not null default 'active' check (status in ('active','archived')),
  archived_at timestamptz,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);
create index if not exists campaigns_owner_idx on public.campaigns(owner_user_id);
create index if not exists campaigns_status_idx on public.campaigns(status);

create table if not exists public.campaign_members (
  id uuid primary key default gen_random_uuid(),
  campaign_id uuid not null references public.campaigns(id) on delete cascade,
  user_id uuid not null references auth.users(id) on delete cascade,
  role text not null check (role in ('dm','player')),
  status text not null default 'active' check (status in ('active','left','removed')),
  joined_at timestamptz not null default now(),
  left_at timestamptz,
  unique (campaign_id, user_id)
);
create index if not exists campaign_members_user_idx on public.campaign_members(user_id);
create index if not exists campaign_members_campaign_status_idx on public.campaign_members(campaign_id,status);
create unique index if not exists one_active_dm_per_campaign_idx on public.campaign_members(campaign_id) where role='dm' and status='active';

create table if not exists public.app_modules (
  id uuid primary key default gen_random_uuid(),
  code text not null unique,
  name text not null,
  description text,
  is_active boolean not null default true,
  created_at timestamptz not null default now()
);
insert into public.app_modules (code,name,description) values
('crafting_gathering','Crafting & Gathering','Crafting, gathering, professions, recipes, materials and timed jobs.'),
('bastions','Bastions','Campaign bastions, facilities, hirelings, projects, turns and events.'),
('relations','Relations','Campaign relationships, affinity, influence and DM-private notes.')
on conflict (code) do update set name=excluded.name, description=excluded.description, is_active=true;

create table if not exists public.campaign_modules (
  campaign_id uuid not null references public.campaigns(id) on delete cascade,
  module_id uuid not null references public.app_modules(id),
  enabled boolean not null default true,
  settings jsonb not null default '{}'::jsonb,
  enabled_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  primary key (campaign_id,module_id)
);

create table if not exists public.content_packs (
  id uuid primary key default gen_random_uuid(),
  code text not null unique,
  name text not null,
  description text,
  access_tier text not null default 'core' check (access_tier in ('core','subscription')),
  source_type text not null default 'original' check (source_type in ('original','open','licensed','partner')),
  source_name text,
  is_active boolean not null default true,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);
insert into public.content_packs (code,name,description,access_tier,source_type)
values ('core','Core Content','Base content available to Free and Premium campaigns.','core','original')
on conflict (code) do update set name=excluded.name, description=excluded.description, access_tier=excluded.access_tier, is_active=true;

create table if not exists public.campaign_content_packs (
  campaign_id uuid not null references public.campaigns(id) on delete cascade,
  content_pack_id uuid not null references public.content_packs(id),
  enabled boolean not null default true,
  enabled_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  primary key (campaign_id,content_pack_id)
);

create table if not exists public.campaign_invitations (
  id uuid primary key default gen_random_uuid(),
  campaign_id uuid not null references public.campaigns(id) on delete cascade,
  kind text not null check (kind in ('email','share_link')),
  email text,
  token_hash text not null unique,
  max_uses integer,
  use_count integer not null default 0 check (use_count >= 0),
  expires_at timestamptz,
  revoked_at timestamptz,
  created_by uuid not null references auth.users(id),
  created_at timestamptz not null default now(),
  check ((kind='email' and email is not null and max_uses=1) or kind='share_link')
);
create index if not exists campaign_invitations_campaign_idx on public.campaign_invitations(campaign_id);

create table if not exists public.characters (
  id uuid primary key default gen_random_uuid(),
  campaign_id uuid not null references public.campaigns(id) on delete cascade,
  owner_membership_id uuid references public.campaign_members(id) on delete set null,
  name text not null check (char_length(trim(name)) between 1 and 120),
  avatar_path text,
  status text not null default 'active' check (status in ('active','archived')),
  archived_at timestamptz,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  unique (campaign_id,name)
);
create index if not exists characters_campaign_idx on public.characters(campaign_id);
create index if not exists characters_owner_membership_idx on public.characters(owner_membership_id);
create unique index if not exists one_active_character_per_membership_idx on public.characters(owner_membership_id) where owner_membership_id is not null and status='active';

create or replace function public.set_updated_at() returns trigger language plpgsql security invoker set search_path=public as $$ begin new.updated_at=now(); return new; end; $$;

DO $$ DECLARE t text; BEGIN
  FOREACH t IN ARRAY ARRAY['profiles','plans','subscriptions','campaigns','campaign_modules','content_packs','campaign_content_packs','characters'] LOOP
    EXECUTE format('drop trigger if exists %I_set_updated_at on public.%I',t,t);
    EXECUTE format('create trigger %I_set_updated_at before update on public.%I for each row execute function public.set_updated_at()',t,t);
  END LOOP;
END $$;

create or replace function public.is_platform_admin() returns boolean language sql stable security definer set search_path=public as $$ select exists(select 1 from public.platform_admins where user_id=auth.uid()); $$;
create or replace function public.is_campaign_member(target_campaign_id uuid) returns boolean language sql stable security definer set search_path=public as $$ select exists(select 1 from public.campaign_members cm join public.campaigns c on c.id=cm.campaign_id where cm.campaign_id=target_campaign_id and cm.user_id=auth.uid() and cm.status='active' and c.status='active'); $$;
create or replace function public.is_campaign_dm(target_campaign_id uuid) returns boolean language sql stable security definer set search_path=public as $$ select public.is_platform_admin() or exists(select 1 from public.campaign_members cm where cm.campaign_id=target_campaign_id and cm.user_id=auth.uid() and cm.role='dm' and cm.status='active'); $$;
create or replace function public.owns_character(target_character_id uuid) returns boolean language sql stable security definer set search_path=public as $$ select exists(select 1 from public.characters ch join public.campaign_members cm on cm.id=ch.owner_membership_id where ch.id=target_character_id and ch.status='active' and cm.user_id=auth.uid() and cm.status='active'); $$;

alter table public.profiles enable row level security;
alter table public.platform_admins enable row level security;
alter table public.plans enable row level security;
alter table public.subscriptions enable row level security;
alter table public.entitlement_overrides enable row level security;
alter table public.campaigns enable row level security;
alter table public.campaign_members enable row level security;
alter table public.app_modules enable row level security;
alter table public.campaign_modules enable row level security;
alter table public.content_packs enable row level security;
alter table public.campaign_content_packs enable row level security;
alter table public.campaign_invitations enable row level security;
alter table public.characters enable row level security;

create policy profiles_select_self_or_admin on public.profiles for select using (user_id=auth.uid() or public.is_platform_admin());
create policy profiles_insert_self on public.profiles for insert with check (user_id=auth.uid() or public.is_platform_admin());
create policy profiles_update_self on public.profiles for update using (user_id=auth.uid() or public.is_platform_admin()) with check (user_id=auth.uid() or public.is_platform_admin());
create policy plans_read_authenticated on public.plans for select using (auth.uid() is not null and is_active);
create policy app_modules_read_authenticated on public.app_modules for select using (auth.uid() is not null and is_active);
create policy content_packs_read_authenticated on public.content_packs for select using (auth.uid() is not null and is_active);
create policy subscriptions_read_self on public.subscriptions for select using (user_id=auth.uid() or public.is_platform_admin());
create policy entitlement_overrides_read_self on public.entitlement_overrides for select using (user_id=auth.uid() or public.is_platform_admin());
create policy campaigns_read_member on public.campaigns for select using (public.is_platform_admin() or public.is_campaign_member(id) or owner_user_id=auth.uid());
create policy campaigns_insert_owner on public.campaigns for insert with check (owner_user_id=auth.uid() or public.is_platform_admin());
create policy campaigns_update_dm on public.campaigns for update using (public.is_campaign_dm(id) or owner_user_id=auth.uid()) with check (public.is_campaign_dm(id) or owner_user_id=auth.uid());
create policy campaigns_delete_dm on public.campaigns for delete using (public.is_platform_admin() or owner_user_id=auth.uid());
create policy campaign_members_read_member on public.campaign_members for select using (public.is_platform_admin() or public.is_campaign_member(campaign_id) or user_id=auth.uid());
create policy campaign_members_insert_dm on public.campaign_members for insert with check (public.is_platform_admin() or public.is_campaign_dm(campaign_id) or (role='dm' and user_id=auth.uid() and exists(select 1 from public.campaigns c where c.id=campaign_id and c.owner_user_id=auth.uid())));
create policy campaign_members_update_dm on public.campaign_members for update using (public.is_campaign_dm(campaign_id) or public.is_platform_admin()) with check (public.is_campaign_dm(campaign_id) or public.is_platform_admin());
create policy campaign_members_delete_dm on public.campaign_members for delete using (public.is_campaign_dm(campaign_id) or public.is_platform_admin());
create policy campaign_modules_read_member on public.campaign_modules for select using (public.is_platform_admin() or public.is_campaign_member(campaign_id));
create policy campaign_modules_manage_dm on public.campaign_modules for all using (public.is_campaign_dm(campaign_id) or public.is_platform_admin()) with check (public.is_campaign_dm(campaign_id) or public.is_platform_admin());
create policy campaign_content_packs_read_member on public.campaign_content_packs for select using (public.is_platform_admin() or public.is_campaign_member(campaign_id));
create policy campaign_content_packs_manage_dm on public.campaign_content_packs for all using (public.is_campaign_dm(campaign_id) or public.is_platform_admin()) with check (public.is_campaign_dm(campaign_id) or public.is_platform_admin());
create policy campaign_invitations_manage_dm on public.campaign_invitations for all using (public.is_campaign_dm(campaign_id) or public.is_platform_admin()) with check (public.is_campaign_dm(campaign_id) or public.is_platform_admin());
create policy characters_read_member on public.characters for select using (public.is_platform_admin() or public.is_campaign_member(campaign_id));
create policy characters_insert_self_or_dm on public.characters for insert with check (public.is_platform_admin() or public.is_campaign_dm(campaign_id) or (owner_membership_id is not null and exists(select 1 from public.campaign_members cm where cm.id=owner_membership_id and cm.campaign_id=campaign_id and cm.user_id=auth.uid() and cm.status='active' and cm.role='player')));
create policy characters_update_owner_or_dm on public.characters for update using (public.is_platform_admin() or public.is_campaign_dm(campaign_id) or public.owns_character(id)) with check (public.is_platform_admin() or public.is_campaign_dm(campaign_id) or public.owns_character(id));
create policy characters_delete_owner_or_dm on public.characters for delete using (public.is_platform_admin() or public.is_campaign_dm(campaign_id) or public.owns_character(id));

commit;
