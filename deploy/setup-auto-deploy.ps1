# One-time Azure setup for automatic deploys from GitHub (.github/workflows/deploy.yml).
# Run from the repo root after `az login`:   & .\deploy\setup-auto-deploy.ps1
# Safe to re-run: each step checks what already exists.
#
# What it does:
#   1. Turns on the web app's persistent /home storage and sets APP_STATE_DIR=/home/data,
#      so briefings, newsletters, the mailing list etc. survive deploys and restarts.
#   2. Copies this computer's saved data files into /home/data (skips files already there,
#      unless you pass -OverwriteData).
#   3. Creates the "github-endo-ecosystem-deploy" app registration that GitHub Actions signs
#      in as (OIDC, no password), trusted only for this repo's main branch.
#   4. Lets it build images in the registry and update this one web app (nothing else).
#   5. Writes its client ID into .github/workflows/deploy.yml.

param([switch]$OverwriteData)

$ErrorActionPreference = 'Stop'
$Sub    = '54814f7d-4d8d-4f2e-895b-f1265fe7d116'
$RG     = 'rg-boldwave-endo-review'
$App    = 'boldwave-endo-review'
$Acr    = 'boldwaveendoacr'
$Repo   = 'jeffreyjbenjamin-bit/endo-ecosystem-poc'
$AppReg = 'github-endo-ecosystem-deploy'
$Scm    = "https://$App.scm.azurewebsites.net"
$StateFiles = 'ai_search_runs.sqlite','rag_briefs.sqlite','comparisons.sqlite',
              'newsletters.sqlite','mailing_list.sqlite','rag_logs.sqlite','rag_logs.jsonl'

function Invoke-Az { $out = & az @args; if ($LASTEXITCODE -ne 0) { throw "az $($args -join ' ') failed" }; $out }

Invoke-Az account set --subscription $Sub

# 1. Persistent storage ---------------------------------------------------------
Write-Host "1/5 Enabling persistent /home storage on $App (restarts the site once)..."
Invoke-Az webapp config appsettings set -g $RG -n $App --output none `
    --settings WEBSITES_ENABLE_APP_SERVICE_STORAGE=true APP_STATE_DIR=/home/data

# 2. Copy saved data -------------------------------------------------------------
Write-Host "2/5 Copying saved data to /home/data..."
$token = Invoke-Az account get-access-token --query accessToken -o tsv
$h = @{ Authorization = "Bearer $token" }
for ($i = 0; $i -lt 20; $i++) {
    try { Invoke-RestMethod -Method Put -Uri "$Scm/api/vfs/data/" -Headers $h | Out-Null; break }
    catch { if ($_.Exception.Response.StatusCode.value__ -eq 409) { break }; Start-Sleep 15 }
}
$existing = @((Invoke-RestMethod -Uri "$Scm/api/vfs/data/" -Headers $h) | ForEach-Object { $_.name })
foreach ($f in $StateFiles) {
    $local = Join-Path 'data' $f
    if (-not (Test-Path $local)) { Write-Host "   skip $f (not on this computer)"; continue }
    if (($existing -contains $f) -and -not $OverwriteData) { Write-Host "   keep $f (already on the server)"; continue }
    Invoke-RestMethod -Method Put -Uri "$Scm/api/vfs/data/$f" -Headers ($h + @{ 'If-Match' = '*' }) `
        -InFile $local -ContentType 'application/octet-stream' | Out-Null
    Write-Host "   uploaded $f"
}

# 3. App registration + OIDC trust ----------------------------------------------
Write-Host "3/5 Setting up the '$AppReg' identity for GitHub Actions..."
$clientId = Invoke-Az ad app list --display-name $AppReg --query '[0].appId' -o tsv
if (-not $clientId) { $clientId = Invoke-Az ad app create --display-name $AppReg --query appId -o tsv }
$spId = Invoke-Az ad sp list --filter "appId eq '$clientId'" --query '[0].id' -o tsv
if (-not $spId) { $spId = Invoke-Az ad sp create --id $clientId --query id -o tsv }

$fedName = 'github-main'
$fed = Invoke-Az ad app federated-credential list --id $clientId --query "[?name=='$fedName'].name" -o tsv
if (-not $fed) {
    $tmp = New-TemporaryFile
    $json = @{ name = $fedName; issuer = 'https://token.actions.githubusercontent.com'
       subject = "repo:${Repo}:ref:refs/heads/main"; audiences = @('api://AzureADTokenExchange')
       description = 'GitHub Actions deploy from main' } | ConvertTo-Json
    [IO.File]::WriteAllText($tmp.FullName, $json, (New-Object Text.UTF8Encoding $false))
    Invoke-Az ad app federated-credential create --id $clientId --parameters "@$tmp" --output none
    Remove-Item $tmp
}

# 4. Least-privilege roles --------------------------------------------------------
Write-Host "4/5 Granting it access to the registry and the web app only..."
$acrId = Invoke-Az acr show -n $Acr --query id -o tsv
$appId = Invoke-Az webapp show -g $RG -n $App --query id -o tsv
foreach ($pair in @(@('Contributor', $acrId), @('Website Contributor', $appId))) {
    $has = Invoke-Az role assignment list --assignee $spId --role $pair[0] --scope $pair[1] --query '[0].id' -o tsv
    if (-not $has) {
        Invoke-Az role assignment create --assignee-object-id $spId --assignee-principal-type ServicePrincipal `
            --role $pair[0] --scope $pair[1] --output none
    }
}

# 5. Workflow config -----------------------------------------------------------------
Write-Host "5/5 Writing client ID into .github/workflows/deploy.yml..."
$wf = Join-Path (Get-Location) '.github/workflows/deploy.yml'
$text = [IO.File]::ReadAllText($wf) -replace 'AZURE_CLIENT_ID: [^\r\n]*', "AZURE_CLIENT_ID: $clientId"
[IO.File]::WriteAllText($wf, $text, (New-Object Text.UTF8Encoding $false))

Write-Host ""
Write-Host "Done. Client ID: $clientId"
Write-Host "Next: commit and push main; the 'Deploy to Azure' workflow will build and deploy."
