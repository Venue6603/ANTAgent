# test_self_improve.ps1
$baseUrl = "http://127.0.0.1:8010"

Write-Host "=== Testing ANTAgent Self-Improvement ===" -ForegroundColor Cyan
Write-Host "Base URL: $baseUrl" -ForegroundColor Gray

# 1. Test status
Write-Host "`n[1] Testing /status endpoint..." -ForegroundColor Yellow
try {
    $status = Invoke-RestMethod -Uri "$baseUrl/status" -Method GET
    Write-Host "Status Response:" -ForegroundColor Green
    $status | Format-List
} catch {
    Write-Host "Status check failed: $_" -ForegroundColor Red
    exit 1
}

# 2. Test OpenAI
Write-Host "`n[2] Testing /debug/openai endpoint..." -ForegroundColor Yellow
try {
    $openai = Invoke-RestMethod -Uri "$baseUrl/debug/openai" -Method GET
    Write-Host "OpenAI Response:" -ForegroundColor Green
    $openai | Format-List
} catch {
    Write-Host "OpenAI debug failed: $_" -ForegroundColor Red
}

# 3. Test code listing
Write-Host "`n[3] Testing /code/list_paths..." -ForegroundColor Yellow
try {
    $paths = Invoke-RestMethod -Uri "$baseUrl/code/list_paths" -Method GET
    Write-Host "Available paths: $($paths.paths.Count) files" -ForegroundColor Green
    $paths.paths | Select-Object -First 5 | ForEach-Object { Write-Host "  - $_" -ForegroundColor Gray }
} catch {
    Write-Host "Code listing failed: $_" -ForegroundColor Red
}

# 4. Test self-improvement
Write-Host "`n[4] Testing /self_improve endpoint..." -ForegroundColor Yellow
$improvementBody = @{
    goal = "Add a comment '# SELF-IMPROVED: $(Get-Date -Format 'yyyy-MM-dd HH:mm')' at the top of AntAgent/app.py file"
    constraints = @{
        paths = @("AntAgent/app.py")
        no_net_new_deps = $true
    }
    rounds = 1
    dry_run = $false  # Set to $true to test without applying
} | ConvertTo-Json -Depth 3

try {
    Write-Host "Sending self-improvement request..." -ForegroundColor Gray
    $result = Invoke-RestMethod -Uri "$baseUrl/self_improve" `
        -Method POST `
        -ContentType "application/json" `
        -Body $improvementBody

    Write-Host "Self-Improvement Result:" -ForegroundColor Green

    foreach ($round in $result.results) {
        Write-Host "`n  Round $($round.round):" -ForegroundColor Cyan
        Write-Host "  - Summary: $($round.summary)" -ForegroundColor White
        Write-Host "  - Applied: $($round.applied)" -ForegroundColor $(if($round.applied){"Green"}else{"Yellow"})
        Write-Host "  - Message: $($round.message)" -ForegroundColor Gray
        if ($round.unified_diff) {
            Write-Host "  - Diff generated: $($round.unified_diff.Length) characters" -ForegroundColor Gray
        }
    }
} catch {
    Write-Host "Self-improve failed: $_" -ForegroundColor Red
    Write-Host "Response: $($_.Exception.Response)" -ForegroundColor Gray
}

Write-Host "`n=== Test Complete ===" -ForegroundColor Cyan