# Models

Model files are NOT committed (see .gitignore) — they're large and derived.
They live at `~/RyzenAI-SW/CNN-examples/object_detection/yolov8m/models/`.

## Regenerate

    cd ~/RyzenAI-SW/CNN-examples/object_detection/yolov8m/models
    wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8m.pt
    python export_to_onnx.py                    # -> yolov8m.onnx

    cd ..
    # INT8 — the exclusion flag is REQUIRED or the model detects nothing
    python quantize_quark.py --input_model_path models/yolov8m.onnx --calib_data_path calib_images --output_model_path models/yolov8m_XINT8.onnx --config XINT8 --exclude_subgraphs "[/model.22/Concat_3], [/model.22/Concat_10]]"

    # INT8 without the exclusion — for the "detects nothing" experiment
    python quantize_quark.py --input_model_path models/yolov8m.onnx --calib_data_path calib_images --output_model_path models/yolov8m_XINT8_no_exclude.onnx --config XINT8

    # BF16 — no exclusion needed; compile takes minutes
    python quantize_quark.py --input_model_path models/yolov8m.onnx --calib_data_path calib_images --output_model_path models/yolov8m_BF16.onnx --config BF16

## ResNet (getting-started example)

The .pt in AMD's repo is a Git LFS pointer (133 bytes). Without git-lfs, download
the real ~95 MB file from GitHub's media endpoint and scp it over, then export:

    python -c "import torch; m = torch.load('models/resnet_trained_for_cifar10.pt', weights_only=False); m.to('cpu'); torch.onnx.export(m, torch.randn(1,3,32,32), 'models/resnet_trained_for_cifar10.onnx', export_params=True, opset_version=17, input_names=['input'], output_names=['output'], dynamic_axes={'input':{0:'batch_size'},'output':{0:'batch_size'}})"

(prepare_model_data.py stalls: it downloads cifar-10-binary.tar.gz from a very slow
UofT host, and that file is only needed for the C++ demo.)