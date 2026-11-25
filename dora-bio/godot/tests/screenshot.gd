extends Node
## Renders the real UI after loading the sample and denoising it, then saves a PNG and quits.
## Needs a display (use xvfb-run on a headless machine) and a real renderer, not --headless:
##   xvfb-run -s "-screen 0 1280x800x24" godot --path . --rendering-driver opengl3 \
##       res://tests/screenshot.tscn -- --out=/absolute/path/screenshot.png

## Loads the sample, denoises it, waits for the frame to be drawn, then saves the viewport as PNG.
func _ready() -> void:
	var out := "user://screenshot.png"
	for arg in OS.get_cmdline_user_args():
		if arg.begins_with("--out="):
			out = arg.trim_prefix("--out=")

	var main: Control = load("res://scenes/Main.tscn").instantiate()
	add_child(main)
	await get_tree().process_frame
	main.load_path(main.SAMPLE_PATH)
	main._on_denoise_pressed()
	await main.bridge.DenoiseFinished
	for i in 5:
		await get_tree().process_frame
	await RenderingServer.frame_post_draw

	var error := get_viewport().get_texture().get_image().save_png(out)
	print("screenshot saved to %s (error %d)" % [out, error])
	get_tree().quit(0 if error == OK else 1)
