"""firesmoke — drone görüntüsünde ateş/duman tespiti, yalnızca ONNX Runtime ile (PyTorch gerekmez).

Kullanım:
    python predict.py --model firesmoke_y11s_640.onnx --source video.mp4 --out sonuc.mp4
    python predict.py --model firesmoke_y11s_640.onnx --source video.mp4 --out sonuc.mp4 --stride 5
    python predict.py --model firesmoke_y11s_640.onnx --source kareler/ --out cikti/
    python predict.py --model firesmoke_y11s_640.onnx --source tek.jpg --out tek_sonuc.jpg

Alarm, zaman filtresiyle verilir: son N karenin en az K'sında aynı sınıf tespit edilirse alarm açılır,
son N karede hiç tespit yoksa kapanır. Tek karelik yanlış tespitler alarma dönüşmez.
Varsayılan eşikler (fire 0.25, smoke 0.30) val setinde kare bazında yanlış alarm ≤ %1 olacak şekilde seçildi.
"""
import argparse
import ast
import json
import re
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

IMG_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
VID_EXT = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
COLORS = {0: (0, 140, 255), 1: (235, 206, 135)}       # BGR: fire turuncu, smoke açık mavi
LABELS_TR = {0: 'ATES', 1: 'DUMAN'}


class Detector:
    """YOLO ONNX çıktısını (1, 4 + nc, N) çözer: letterbox → model → sınıf başına eşik → NMS → orijinal koordinat."""

    def __init__(self, model_path, conf=None, iou=0.5, providers=None):
        avail = ort.get_available_providers()
        self.sess = ort.InferenceSession(str(model_path), providers=providers or avail)
        inp = self.sess.get_inputs()[0]
        self.inp_name, (self.h, self.w) = inp.name, inp.shape[2:4]
        meta = self.sess.get_modelmeta().custom_metadata_map
        self.names = {int(k): v for k, v in ast.literal_eval(meta.get('names', "{0: 'fire', 1: 'smoke'}")).items()}
        self.conf = {0: 0.25, 1: 0.30} if conf is None else dict(conf)
        self.iou = iou

    def _letterbox(self, img):
        H, W = img.shape[:2]
        r = min(self.h / H, self.w / W)
        nh, nw = round(H * r), round(W * r)
        top, left = (self.h - nh) // 2, (self.w - nw) // 2
        canvas = np.full((self.h, self.w, 3), 114, np.uint8)
        canvas[top:top + nh, left:left + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        x = canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0   # BGR→RGB, HWC→NCHW
        return np.ascontiguousarray(x), r, left, top

    def __call__(self, img):
        """img: BGR (OpenCV). Dönen: [(sınıf, güven, x1, y1, x2, y2), ...] orijinal piksel koordinatında."""
        x, r, left, top = self._letterbox(img)
        out = self.sess.run(None, {self.inp_name: x})[0][0].T          # (N, 4 + nc)
        scores = out[:, 4:]
        cls = scores.argmax(1)
        conf = scores[np.arange(len(cls)), cls]
        thr = np.array([self.conf.get(int(c), 1.0) for c in cls])
        keep = conf >= thr
        if not keep.any():
            return []
        b, cls, conf = out[keep, :4], cls[keep], conf[keep]
        xywh = np.stack([b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2, b[:, 2], b[:, 3]], 1)
        dets = []
        for c in np.unique(cls):                                         # sınıf başına NMS
            m = np.where(cls == c)[0]
            idx = cv2.dnn.NMSBoxes(xywh[m].tolist(), conf[m].tolist(), 0.0, self.iou)
            for i in np.array(idx).reshape(-1):
                j = m[i]
                x1, y1, w, h = xywh[j]
                H, W = img.shape[:2]
                bx = np.array([(x1 - left) / r, (y1 - top) / r, (x1 + w - left) / r, (y1 + h - top) / r])
                bx = np.clip(bx, 0, [W, H, W, H])
                dets.append((int(c), float(conf[j]), *map(float, bx)))
        return sorted(dets, key=lambda d: -d[1])


class AlarmFilter:
    """Son n karenin en az k'sında tespit → alarm açılır; son n karede hiç tespit yok → kapanır."""

    def __init__(self, k=3, n=5):
        self.k, self.hist, self.on = k, deque(maxlen=n), False

    def update(self, detected):
        self.hist.append(bool(detected))
        if not self.on and sum(self.hist) >= self.k:
            self.on = True
        elif self.on and not any(self.hist):
            self.on = False
        return self.on


def draw(img, dets, alarms, names, info=''):
    out = img.copy()
    t = max(1, round(max(img.shape[:2]) / 700))
    for c, s, x1, y1, x2, y2 in dets:
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), COLORS.get(c, (0, 0, 255)), 2 * t)
        cv2.putText(out, f'{names.get(c, c)} {s:.2f}', (int(x1), max(int(y1) - 4 * t, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45 * t, COLORS.get(c, (0, 0, 255)), t, cv2.LINE_AA)
    active = [LABELS_TR[c] for c in sorted(alarms) if alarms[c]]
    bar_h = 26 * t
    cv2.rectangle(out, (0, 0), (out.shape[1], bar_h), (0, 0, 200) if active else (40, 40, 40), -1)
    text = ('ALARM: ' + ' + '.join(active)) if active else 'normal'
    cv2.putText(out, text, (8 * t, 19 * t), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * t, (255, 255, 255), t + 1, cv2.LINE_AA)
    if info:
        (tw, _), _ = cv2.getTextSize(info, cv2.FONT_HERSHEY_SIMPLEX, 0.45 * t, t)
        cv2.putText(out, info, (out.shape[1] - tw - 8 * t, 18 * t), cv2.FONT_HERSHEY_SIMPLEX, 0.45 * t,
                    (220, 220, 220), t, cv2.LINE_AA)
    return out


def natural_key(p):
    # 'frame7' < 'frame21' (sözlük sırası değil, sayı sırası)
    return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', p.name)]


def frames_from(source, stride=1):
    """(ad, kare, saniye) üretir. stride > 1 ise her stride'ıncı kare işlenir (videoda aradakiler çözülmeden atlanır)."""
    src = Path(source)
    if not src.exists():
        raise SystemExit(f'Kaynak bulunamadı: {src.resolve()}  (dosya adını ve klasörü kontrol et)')
    if src.is_dir():
        files = sorted((q for q in src.iterdir() if q.suffix.lower() in IMG_EXT), key=natural_key)
        for p in files[::stride]:
            yield p.name, cv2.imread(str(p)), None
    elif src.suffix.lower() in VID_EXT:
        cap = cv2.VideoCapture(str(src))
        if not cap.isOpened() or cap.get(cv2.CAP_PROP_FRAME_COUNT) == 0:
            raise SystemExit(f'Video açılamadı: {src.name} — OpenCV bu codec\'i (ör. AV1) çözemiyor olabilir; '
                             'H.264 bir sürüm dene.')
        fps, i = cap.get(cv2.CAP_PROP_FPS) or 0, 0
        while True:
            if i % stride:
                if not cap.grab():                                      # atla: çözmeden geç (hızlı)
                    break
                i += 1
                continue
            ok, f = cap.read()
            if not ok:
                break
            yield f'frame{i:06d}', f, (i / fps if fps else None)
            i += 1
        cap.release()
    else:
        yield src.name, cv2.imread(str(src)), None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', required=True)
    ap.add_argument('--source', required=True, help='görüntü, klasör ya da video')
    ap.add_argument('--out', required=True, help='çıktı: .mp4 (video), klasör ya da tek görüntü')
    ap.add_argument('--conf-fire', type=float, default=0.25)
    ap.add_argument('--conf-smoke', type=float, default=0.30)
    ap.add_argument('--k', type=int, default=3, help='alarm için son N karenin kaçında tespit olmalı')
    ap.add_argument('--n', type=int, default=5, help='zaman penceresi (kare)')
    ap.add_argument('--fps', type=float, default=None, help='çıktı videosunun FPS değeri (varsayılan: kaynak FPS / stride)')
    ap.add_argument('--stride', type=int, default=1,
                    help='her kaçıncı kare işlensin (ör. 30 FPS videoda 5 → saniyede 6 kare; ~5 kat hızlı). '
                         'K/N işlenen kareler üzerinden sayılır.')
    ap.add_argument('--out-width', type=int, default=1280,
                    help='çıktı videosu/görüntüsünün en büyük genişliği (tespit tam çözünürlükte yapılır; 0 = küçültme)')
    a = ap.parse_args()

    det = Detector(a.model, conf={0: a.conf_fire, 1: a.conf_smoke})
    filt = {c: AlarmFilter(a.k, a.n) for c in det.names}
    out = Path(a.out)
    src_is_video = Path(a.source).suffix.lower() in VID_EXT
    stride = max(1, a.stride)
    src_fps = cv2.VideoCapture(a.source).get(cv2.CAP_PROP_FPS) if src_is_video else 0
    fps = a.fps or (src_fps / stride if src_fps else 5)
    total = int(cv2.VideoCapture(a.source).get(cv2.CAP_PROP_FRAME_COUNT)) // stride if src_is_video else None
    writer, log, t0 = None, [], time.time()
    for k, (name, img, sec) in enumerate(frames_from(a.source, stride)):
        if img is None:
            continue
        d = det(img)
        alarms = {c: filt[c].update(any(x[0] == c for x in d)) for c in filt}
        vis = draw(img, d, alarms, det.names, info=name if sec is None else f'{name}  {sec:6.1f} sn')
        if a.out_width and vis.shape[1] > a.out_width:
            vis = cv2.resize(vis, (a.out_width, int(round(vis.shape[0] * a.out_width / vis.shape[1] / 2)) * 2),
                             interpolation=cv2.INTER_AREA)
        log.append({'kare': name, 'sn': None if sec is None else round(sec, 2),
                    'tespit': [[det.names[x[0]], round(x[1], 3)] for x in d],
                    'alarm': [det.names[c] for c, v in alarms.items() if v]})
        if total and (k + 1) % 100 == 0:
            el = time.time() - t0
            print(f'  {k + 1}/{total} kare · {el:.0f} sn · kalan ~{el / (k + 1) * (total - k - 1):.0f} sn', flush=True)
        if out.suffix.lower() in VID_EXT:
            if writer is None:
                out.parent.mkdir(parents=True, exist_ok=True)
                size = (vis.shape[1], vis.shape[0])
                writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
            if (vis.shape[1], vis.shape[0]) != size:                     # klasördeki kareler farklı boyutta olabilir
                vis = cv2.resize(vis, size)
            writer.write(vis)
        elif out.suffix.lower() in IMG_EXT:
            out.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out), vis)
        else:
            out.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out / f'{Path(name).stem}.jpg'), vis)
    if writer is not None:
        writer.release()
    if not log:
        raise SystemExit('Hiç kare okunamadı — kaynak boş ya da okunamıyor.')
    dt = time.time() - t0
    first = next((r for r in log if r['alarm']), None)
    n_al = sum(bool(r['alarm']) for r in log)
    summary = {'kare': len(log), 'stride': stride, 'sure_sn': round(dt, 1),
               'kare_basina_ms': round(1000 * dt / max(len(log), 1), 1),
               'alarmli_kare': n_al, 'alarmli_kare_%': round(100 * n_al / len(log), 1),
               'ilk_alarm': None if first is None else (first['kare'] if first['sn'] is None else f"{first['kare']} ({first['sn']} sn)")}
    (out.with_suffix('.json') if out.suffix else out / 'log.json').write_text(
        json.dumps({'ozet': summary, 'kareler': log}, ensure_ascii=False, indent=1))
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
