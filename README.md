# Yolov8n para semaforos de pedestres

## Como testar em um video na Raspberry PI 5

1. Exportar os arquivos "best.onnx", "infer_rpi.py" e o video para teste
2. Instalar as seguintes dependencias:
```
sudo apt-get update
sudo apt-get install -y python3-opencv
```
3. Executar o seguinte comando:
```
python3 infer_rpi.py --onnx best.onnx --source video_teste.mp4 --imgsz 640 --conf 0.45 --iou 0.60
```
