extends Node
## Headless end-to-end check of the Godot shell. Build the C# assembly, then run:
##   cd godot && dotnet build && godot --headless --path . res://tests/smoke_test.tscn
## Loads Main.tscn, loads the bundled sample, checks its noise, denoises it, runs a short adaptation, and exits
## with code 0 on success or 1 on failure. A watchdog fails the run if any step hangs.

const TIMEOUT_SECONDS := 180.0

## Set by the first failure, so later checks and the watchdog don't report twice.
var _failed := false


## Runs the whole scenario. Each step awaits the bridge's real signals, exactly as the UI would.
func _ready() -> void:
	get_tree().create_timer(TIMEOUT_SECONDS).timeout.connect(func(): _fail("timed out"))

	var main: Control = load("res://scenes/Main.tscn").instantiate()
	add_child(main)
	await get_tree().process_frame
	var bridge: Node = main.bridge

	if not _check(bridge != null and bridge.has_signal("DenoiseFinished"), "C# bridge is attached and exposes its signals"): return
	if not _check(not bridge.IsBusy(), "bridge starts idle"): return
	print("model: ", bridge.DescribeModel())
	if not _check(bridge.DescribeModel().contains("models/dora_bio_weights.bin") or bridge.DescribeModel().contains("user://"), "a trained model was loaded"): return

	main.load_path(main.SAMPLE_PATH)
	var raw: Texture2D = main.raw_view.texture
	if not _check(raw != null, "raw texture shown after loading the sample"): return
	if not _check(raw.get_image().get_format() == Image.FORMAT_RF, "raw image stays 32-bit float (FORMAT_RF)"): return
	print("noise check: ", bridge.DescribeNoise())
	if not _check(bridge.DescribeNoise().contains("independent"), "noise check ran on load (the synthetic sample's noise is independent)"): return

	main._on_denoise_pressed()
	if not _check(bridge.IsBusy(), "denoise runs as a background job"): return
	var result: Array = await bridge.DenoiseFinished
	await get_tree().process_frame
	var denoised: Image = result[0]
	if not _check(denoised.get_width() == raw.get_width() and denoised.get_height() == raw.get_height(), "denoised image has the input's size"): return
	if not _check(main.dora_view.texture != null and main.baseline_view.texture != null, "DORA and baseline panels filled"): return
	var noise_raw: float = result[2]
	var noise_dora: float = result[3]
	print("estimated noise sigma: raw %.4f -> DORA %.4f (baseline %.4f), %.2fs" % [noise_raw, noise_dora, result[4], result[5]])
	if not _check(noise_dora < noise_raw * 0.7, "DORA reduces the estimated noise by more than 30%"): return

	bridge.StartAdapt(3)
	var adapt: Array = await bridge.AdaptFinished
	if not _check(adapt[0] == 3, "adaptation ran 3 steps"): return
	if not _check(bridge.HasUnsavedChanges(), "adaptation marks weights unsaved (nothing written implicitly)"): return
	await bridge.DenoiseFinished   # the UI re-denoises after adapting
	await get_tree().process_frame
	if not _check(not bridge.IsBusy(), "bridge is idle again"): return

	print("SMOKE TEST PASSED")
	get_tree().quit(0)


## Prints "ok: …" and returns true, or fails the run. Use as `if not _check(...): return`.
func _check(condition: bool, what: String) -> bool:
	if _failed:
		return false
	if condition:
		print("  ok: ", what)
		return true
	_fail(what)
	return false


## Reports the failure once and quits with exit code 1 (what bricks.sh gate 8 looks for).
func _fail(what: String) -> void:
	if _failed:
		return
	_failed = true
	printerr("SMOKE TEST FAILED: ", what)
	get_tree().quit(1)
