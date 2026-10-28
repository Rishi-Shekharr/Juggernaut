param(
    [ValidateSet('os','o2','o2-iram')][string[]]$Variants=@('os','o2','o2-iram'),
    [ValidateRange(1,8)][int]$Jobs=2,
    [ValidateSet('dio','qio')][string]$FlashMode='dio',
    [ValidateSet('40','80')][string]$FlashFrequency='40',
    [ValidateSet('fp32','layer1-int8')][string]$Experiment='fp32',
    [string]$ArduinoCli='C:\Program Files\Arduino IDE\resources\app\lib\backend\resources\arduino-cli.exe'
)
$ErrorActionPreference='Stop'
$root=$PSScriptRoot
if (!(Test-Path -LiteralPath $ArduinoCli)) { throw "Arduino CLI not found: $ArduinoCli" }
# Explicit flash settings isolate compiler/kernel comparisons. DIO 40 MHz is
# the conservative default; alternate settings get separate artifact folders.
$selectedFlashMode=$FlashMode.ToLowerInvariant()
$selectedFlashFrequencyMHz=[int]$FlashFrequency
$fqbn="esp32:esp32:esp32:CPUFreq=240,FlashSize=4M,PartitionScheme=huge_app,FlashMode=$selectedFlashMode,FlashFreq=$selectedFlashFrequencyMHz"
$sources=@('policy_benchmark\policy_benchmark.ino','policy_benchmark\policy_kernels.h','policy_benchmark\policy_data.h','policy_benchmark\policy_int8_weight.h','reference.json')
if (Test-Path -LiteralPath (Join-Path $root 'policy_benchmark\policy_quantized_data.h')) {
    $sources += 'policy_benchmark\policy_quantized_data.h'
}
if ($Experiment -eq 'layer1-int8') {
    $sources += 'quantization_reference.json'
    $buildRoot=Join-Path $root 'quantized_build'
    $binaryRoot=Join-Path $root 'quantized_binaries'
} else {
    $buildRoot=Join-Path $root 'optimized_build'
    $binaryRoot=Join-Path $root 'optimized_binaries'
}
$hashes=[ordered]@{}
foreach ($source in $sources) {
    $hashes[$source]=(Get-FileHash -LiteralPath (Join-Path $root $source) -Algorithm SHA256).Hash.ToLowerInvariant()
}
$reference=Get-Content -LiteralPath (Join-Path $root 'reference.json') -Raw | ConvertFrom-Json
$cliVersion=(& $ArduinoCli version | Out-String).Trim()
foreach ($variant in ($Variants | Select-Object -Unique)) {
    $optimization=if ($variant -eq 'os') {'-Os'} else {'-O2'}
    $optId=if ($variant -eq 'os') {1} else {2}
    $iram=if ($variant -eq 'o2-iram') {1} else {0}
    $digest=[System.Security.Cryptography.SHA256]::Create()
    try {
        $identityText=($hashes.Values -join ':')+":${variant}:${Experiment}:$fqbn"
        $identityBytes=$digest.ComputeHash([System.Text.Encoding]::UTF8.GetBytes($identityText))
        $identity=([BitConverter]::ToString($identityBytes)).Replace('-','').ToLowerInvariant()
    } finally { $digest.Dispose() }
    $idPrefix=if ($Experiment -eq 'layer1-int8') {'int8w_l1int8_'} else {'int8w_'}
    $buildId=$idPrefix+$identity.Substring(0,16)+'_'+$variant.Replace('-','_')
    if ($selectedFlashMode -eq 'dio' -and $selectedFlashFrequencyMHz -eq 40) {
        $buildPath=Join-Path $buildRoot $variant
        $outputPath=Join-Path $binaryRoot $variant
    } else {
        $settingsGroup="${selectedFlashMode}${selectedFlashFrequencyMHz}"
        $buildPath=Join-Path $buildRoot "$settingsGroup\$variant"
        $outputPath=Join-Path $binaryRoot "$settingsGroup\$variant"
    }
    New-Item -ItemType Directory -Path $buildPath,$outputPath -Force | Out-Null
    $logPath=Join-Path $outputPath 'build.log'
    $metadataPath=Join-Path $outputPath 'build-info.json'
    $metadata=[ordered]@{
        schema_version=1; variant=$variant; build_id=$buildId; status='building'; utc_started=[DateTime]::UtcNow.ToString('o')
        fqbn=$fqbn; optimization=$optimization; dense_iram=($iram -eq 1)
        experiment='first-layer-weight-only-int8'; default_quantization='L0_WEIGHT_INT8'
        flash_mode_requested=$selectedFlashMode; flash_frequency_requested_mhz=$selectedFlashFrequencyMHz
        flash_frequency_requested_hz=($selectedFlashFrequencyMHz * 1000000)
        arduino_cli=$cliVersion; policy_sha256=$reference.checkpoint_sha256
        source_sha256=$hashes; hardware_tested=$false
    }
    $metadata | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $metadataPath -Encoding UTF8
    $arguments=@('--config-file',(Join-Path $root 'arduino-cli.yaml'),'compile',
        '--fqbn',$fqbn,'--jobs',"$Jobs",'--warnings','default','--verbose',
        '--build-path',$buildPath,'--output-dir',$outputPath,
        '--build-property',"compiler.optimization_flags=$optimization",
        '--build-property',"compiler.cpp.extra_flags=-DPOLICY_OPT_LEVEL=$optId -DPOLICY_DENSE_IRAM=$iram -DPOLICY_BUILD_ID=$buildId",
        (Join-Path $root 'policy_benchmark'))
    Write-Output "Building $variant ($optimization; dense IRAM=$iram). Log: $logPath"
    $previousPreference=$ErrorActionPreference
    try {
        # Windows PowerShell 5.1 otherwise treats native stderr as a terminating
        # error before the compiler exit code can be inspected.
        $ErrorActionPreference='Continue'
        & $ArduinoCli @arguments > $logPath 2>&1
        $compileExit=$LASTEXITCODE
    } finally { $ErrorActionPreference=$previousPreference }
    $metadata.utc_finished=[DateTime]::UtcNow.ToString('o')
    $metadata.exit_code=$compileExit
    if ($compileExit -ne 0) {
        $metadata.status='failed'
        $metadata | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $metadataPath -Encoding UTF8
        Get-Content -LiteralPath $logPath -Tail 35 | Write-Output
        throw "Build failed for $variant (exit $compileExit). See $logPath"
    }
    foreach ($source in $sources) {
        $afterHash=(Get-FileHash -LiteralPath (Join-Path $root $source) -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($afterHash -ne $hashes[$source]) {
            $metadata.status='source_changed_during_build'
            $metadata | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $metadataPath -Encoding UTF8
            throw "Source changed during build: $source. Rebuild before uploading."
        }
    }
    $metadata.status='compiled'
    $metadata.binaries=@(Get-ChildItem -LiteralPath $outputPath -Filter '*.bin' | ForEach-Object {
        [ordered]@{name=$_.Name;bytes=$_.Length;sha256=(Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant()}
    })
    $metadata | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $metadataPath -Encoding UTF8
    Get-Content -LiteralPath $logPath | Select-String -Pattern '^Sketch uses','^Global variables use' | ForEach-Object { $_.Line }
    Write-Output "Compiled $variant. No firmware was uploaded."
}
