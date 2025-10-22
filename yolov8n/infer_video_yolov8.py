# infer_video_yolov8.py
import cv2, time, argparse, os
from ultralytics import YOLO

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="caminho do .pt (ex.: runs/detect/train/weights/best.pt)")
    ap.add_argument("--source", required=True, help="caminho do vídeo (ex.: C:/.../video.mp4)")
    ap.add_argument("--out", default="", help="arquivo de saída (ex.: C:/.../saida.mp4). Vazio = não salvar")
    ap.add_argument("--imgsz", type=int, default=640, help="tamanho de inferência (pode ser maior que o treino)")
    ap.add_argument("--conf", type=float, default=0.25, help="limiar de confiança")
    ap.add_argument("--device", default=0, help="0 ou cpu")
    ap.add_argument("--show", action="store_true", help="mostra janela com preview")
    return ap.parse_args()

def main():
    args = parse_args()

    # carrega modelo
    model = YOLO(args.model)

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise SystemExit(f"Não consegui abrir o vídeo: {args.source}")

    # leitura de propriedades do vídeo
    fps_in  = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w_in    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_in    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # writer (opcional)
    writer = None
    if args.out:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # ou 'XVID'
        writer = cv2.VideoWriter(args.out, fourcc, fps_in, (w_in, h_in))
        if not writer.isOpened():
            print("[WARN] Não consegui abrir o VideoWriter, saída não será salva.")
            writer = None

    # loop
    avg_fps, alpha = 0.0, 0.1  # média móvel para FPS
    while True:
        ok, frame = cap.read()
        if not ok: break

        t0 = time.perf_counter()
        # inferência: você pode passar o frame (np.ndarray) direto
        results = model.predict(
            source=frame,           # frame único
            imgsz=args.imgsz,
            conf=args.conf,
            device=args.device,
            verbose=False
        )
        # desenha anotações usando o utilitário da Ultralytics
        annotated = results[0].plot()  # retorna uma cópia do frame com caixas/labels

        # FPS
        dt = time.perf_counter() - t0
        inst_fps = 1.0 / max(dt, 1e-6)
        avg_fps = (1 - alpha) * avg_fps + alpha * inst_fps if avg_fps > 0 else inst_fps
        cv2.putText(annotated, f"FPS: {avg_fps:.1f}", (12, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)

        if args.show:
            cv2.imshow("YOLOv8 - Preview", annotated)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC para sair
                break

        if writer is not None:
            writer.write(annotated)

    cap.release()
    if writer: writer.release()
    cv2.destroyAllWindows()
    print("✅ Terminado.")
    if args.out:
        print(f"Vídeo salvo em: {os.path.abspath(args.out)}")

if __name__ == "__main__":
    main()
