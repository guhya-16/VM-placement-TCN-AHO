param (
    [string]$ExampleName = "BasicFirstExample"
)

$javaPath = "C:\Program Files\Java\jdk-24\bin"
$cloudsimDir = "$PSScriptRoot\cloudsim"
if (-not (Test-Path $cloudsimDir)) {
    $cloudsimDir = "$PSScriptRoot\cloudsimplus-examples"
}
if (-not (Test-Path $cloudsimDir)) {
    $cloudsimDir = "$PSScriptRoot\vm-placement-tcn-aho\cloudsim"
}
if (-not (Test-Path $cloudsimDir)) {
    $cloudsimDir = "$PSScriptRoot"
}

$repoPath = "$HOME\.m2\repository"
$jars = @(Get-ChildItem -Recurse -Filter *.jar $repoPath | Select-Object -ExpandProperty FullName) -join ";"

$classesDir = "$cloudsimDir\target\classes"
if (-not (Test-Path $classesDir)) {
    New-Item -ItemType Directory -Force -Path $classesDir | Out-Null
}

$searchFile = Get-ChildItem -Recurse -Filter "$ExampleName.java" "$cloudsimDir\src\main\java" | Select-Object -First 1
if (-not $searchFile) {
    Write-Error "Could not find $ExampleName.java in $cloudsimDir\src\main\java"
    exit 1
}

Write-Host "Compiling $($searchFile.FullName)..." -ForegroundColor Cyan
& "$javaPath\javac.exe" -cp $jars -d $classesDir $searchFile.FullName

if ($LASTEXITCODE -eq 0) {
    Write-Host "Running $ExampleName..." -ForegroundColor Green
    
    # Derive fully qualified class name from relative path
    $rel = $searchFile.FullName.Substring(($cloudsimDir + "\src\main\java\").Length)
    $className = ($rel -replace "\.java$", "") -replace "\\", "."
    
    & "$javaPath\java.exe" -cp "$classesDir;$jars" $className
} else {
    Write-Host "Compilation failed with exit code $LASTEXITCODE" -ForegroundColor Red
}
