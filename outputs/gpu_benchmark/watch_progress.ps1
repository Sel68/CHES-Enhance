$Host.UI.RawUI.WindowTitle = 'CHES GPU benchmark progress'
Get-Content -LiteralPath (Join-Path $PSScriptRoot 'progress.log') -Wait
