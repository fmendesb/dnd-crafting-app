# V2 RLS Security Checklist

Run these checks after any meaningful schema, RLS, invitation, character, module, or entitlement change. Tests should use transactions and roll back temporary rows.

## Authentication and secrets
- Passwords are handled only by Supabase Auth and are stored as bcrypt hashes, never plaintext.
- Client code uses only a Supabase publishable/anon key. No `service_role`, `sb_secret`, database password, or other elevated secret is committed to GitHub or exposed to the client.
- Auth email confirmation remains enabled.

## Anonymous user
- Cannot read campaigns, campaign memberships, characters, invitations, legacy state, or platform-admin data.
- Has no direct table privileges on private application data.

## Unrelated authenticated user
- Cannot read a campaign they do not belong to.
- Cannot read that campaign's members, characters, modules, invitations, or module state.
- Cannot update or delete unrelated campaign data.

## Player
- Can read their active campaign and intended party-visible data.
- Cannot read campaign invitations.
- Can update their own character's permitted profile fields.
- Cannot update another character.
- Cannot update campaign settings, module activation, or campaign memberships.
- Cannot insert their character into another campaign.
- Cannot change their character's `campaign_id` or `owner_membership_id`.
- Cannot create a second active character in the same campaign.
- Cannot bypass campaign/module entitlement limits.

## Dungeon Master
- Can manage their own campaign settings, modules, invitations, and character/module overrides allowed by product rules.
- Cannot manage an unrelated campaign.
- Cannot accidentally demote themselves by accepting their own player invitation.
- Free-plan DM cannot create a second campaign or enable more modules than allowed.

## Invitations
- Raw invitation secrets are never stored; only SHA-256 hashes are persisted.
- Revoked invitations fail.
- Expired invitations fail.
- Single-use email invitations cannot be reused.
- Email-targeted invitations reject a different authenticated email.
- Share invitations respect campaign participation limits.
- Already-active player acceptance is idempotent.

## Global/admin data
- Players cannot modify plans, professions, global content packs, subscriptions, entitlement overrides, or platform-admin membership.
- `auth.users` is not readable by anon/authenticated application roles.
- `platform_admins` has no client table access; admin checks go through controlled private helpers.

## SECURITY DEFINER RPCs
Review every authenticated-callable `SECURITY DEFINER` function manually. It must:
- verify `auth.uid()`;
- validate object membership/ownership and campaign state;
- enforce applicable entitlement/business rules;
- use a fixed safe `search_path`;
- avoid accepting caller-controlled identifiers without authorization checks;
- expose EXECUTE only to the minimum required role.

Current intentional RPCs:
- `public.accept_campaign_invitation(text)`
- `public.create_my_character(uuid,text,uuid,uuid)`

## Production-readiness checks still required
- Enable leaked-password protection when the Supabase plan supports it.
- Configure production password-strength rules.
- Add CAPTCHA/rate-limit protections before public signup.
- Configure custom SMTP for production transactional auth emails.
- Protect Supabase/GitHub owner accounts with MFA.
- Review SSL/network restrictions, backups, monitoring, privacy, account deletion, and incident-response procedures.
- Re-run Supabase Security Advisor and Performance Advisor before every release candidate.

## Crafting & Gathering integrity rule
Once gameplay state is rebuilt, do not trust direct client updates for XP, inventory, crafting, gathering, trades, recipe discovery, timers, or other economically meaningful state. Route these through validated transactional server/database functions so a player cannot forge progression by crafting their own API request.
