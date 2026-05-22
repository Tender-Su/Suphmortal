# RTK - Rust Token Killer (Codex CLI)

**Usage**: Token-optimized CLI proxy for shell commands.

## Rule

Use `rtk` for external commands that may produce long output, especially `git`, `cargo`, `rg`, `npm`, and test runners. Do not wrap PowerShell built-ins, environment-variable probes, pipelines, or short shell expressions with `rtk`; commands such as `Get-Location`, `Get-Content`, `Test-Path`, `$env:...`, and `Select-String` should run directly.

Examples:

```bash
rtk git status
rtk cargo test
rtk npm run build
rtk pytest -q
```

Run these directly:

```powershell
Get-Location
Get-Content .\AGENTS.md
$env:PYO3_PYTHON
Test-Path .\mortal\config.toml
```

## Meta Commands

```bash
rtk gain            # Token savings analytics
rtk gain --history  # Recent command savings history
rtk proxy <cmd>     # Run raw command without filtering
```

## Verification

```bash
rtk --version
rtk gain
which rtk
```
