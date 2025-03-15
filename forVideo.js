import {
  PoseLandmarker,
  FilesetResolver,
  DrawingUtils
} from "https://cdn.skypack.dev/@mediapipe/tasks-vision@0.10.0";

// DOM 要素の取得
const fileInput = document.getElementById('fileInput');
const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const startButton = document.getElementById('startButton');
const stopButton = document.getElementById('stopButton');
const downloadLink = document.getElementById('downloadLink');
const downloadJSONLink = document.getElementById('downloadJSONLink');
const analyzeVideoJsonLink = document.getElementById('analyzeVideoJsonLink');
const loadingTxt = document.getElementById('loadingTxt');

// Canvas コンテキストや MediaRecorder 関連の変数
let ctx = canvas.getContext('2d');
let mediaRecorder;
let recordedChunks = [];
let stream; // canvas.captureStream() から取得する MediaStream

// PoseLandmarker 関連
let poseLandmarker = null;
let drawingUtils = null;
let runningMode = 'VIDEO'; // 初期モード ("IMAGE" / "VIDEO")
let lastVideoTime = -1;

// 保存用のJSONデータ
let poseJSON = [];

/**
 * ファイル名を安全な文字列に変換する（サニタイズ）
 */
function sanitizeFilename(filename) {
  // 英数字とハイフン・アンダースコア・ドット以外はアンダースコアに置換
  return filename.replace(/[^a-zA-Z0-9\-_.]/g, '_');
}

/**
 * PoseLandmarker を初期化（非同期）
 */
async function initPoseLandmarker() {
  // WASM アセットをロード
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm"
  );
  // PoseLandmarker インスタンスを作成
  poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
      delegate: "GPU"
    },
    runningMode: runningMode,
    numPoses: 2
  });
  console.log("PoseLandmarker initialized");
}

// PoseLandmarker の読み込みを開始
initPoseLandmarker().then(() => {
  console.log("PoseLandmarker is ready");
  fileInput.disabled = false;
  loadingTxt.style.display = 'none';
});

/**
 * ファイルが選択されたときの処理
 */
fileInput.addEventListener('change', () => {
  const file = fileInput.files[0];
  if (!file) return;

  // 選択された動画ファイルを video.src に設定
  const fileURL = URL.createObjectURL(file);
  video.src = fileURL;
  video.style.display = 'block';

  // メタデータ読み込み完了後に Canvas サイズを動画に合わせる
  video.addEventListener('loadedmetadata', () => {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    drawingUtils = new DrawingUtils(ctx); // DrawingUtils の初期化
  }, { once: true });

  // 録画開始ボタンを有効に
  startButton.disabled = false;
});

/**
 * 動画再生開始 -> 毎フレーム描画
 */
video.addEventListener('play', () => {
  poseJSON = []; // 初期化
  // モデルがまだ読み込まれていなくてもとりあえず実行開始
  requestAnimationFrame(drawFrame);
});

/**
 * 毎フレーム、Video を Canvas に描画し、PoseLandmarker で姿勢推定
 */
async function drawFrame() {
  if (video.paused || video.ended) {
    return;
  }

  // 動画フレームを Canvas に描画
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  // PoseLandmarker が準備できていれば、姿勢推定を実行
  if (poseLandmarker) {
    const startTimeMs = performance.now();

    // 前回から再生位置が進んでいれば姿勢推定
    if (lastVideoTime !== video.currentTime) {
      lastVideoTime = video.currentTime;
      poseLandmarker.detectForVideo(video, startTimeMs, (result) => {
        ctx.save();
        // 既に動画フレームが描画されているので clearRect は行わない

        // 推定結果を JSON に保存
        poseJSON.push(result);

        // 推定した各ポーズのランドマークを描画
        for (const landmarks of result.landmarks) {
          drawingUtils.drawLandmarks(landmarks, {
            radius: (data) =>
              data.from
                ? DrawingUtils.lerp(data.from.z, -0.15, 0.1, 5, 1)
                : 1,
          });
          drawingUtils.drawConnectors(landmarks, PoseLandmarker.POSE_CONNECTIONS);
        }
        ctx.restore();
      });
    }
  }

  // 次フレームを要求
  requestAnimationFrame(drawFrame);
}

/**
 * 録画開始ボタンが押された
 */
startButton.addEventListener('click', () => {
  // 動画を再生
  if (video.paused) {
    video.play();
  }

  // Canvas から MediaStream を取得
  stream = canvas.captureStream(30);

  // IOSはWEBMを対応してない！
  // だから新しいMediaRecorderを作るときに"mimeType"をvideo/mp4に変える必要がある。
  // MediaRecorder を作成
  if (MediaRecorder.isTypeSupported('video/mp4')) {
    mediaRecorder = new MediaRecorder(stream, {
      mimeType: 'video/mp4', videoBitsPerSecond: 100000
    });
  } else if (MediaRecorder.isTypeSupported('video/webm; codecs=vp9')) {
    mediaRecorder = new MediaRecorder(stream, {
      mimeType: 'video/webm; codecs=vp9'
    });
  } else if (MediaRecorder.isTypeSupported('video/webm')) {
    mediaRecorder = new MediaRecorder(stream, {
      mimeType: 'video/webm'
    });
  } else {
    console.error("no suitable mimetype found for this device");
  }
  recordedChunks = [];

  // 録画中のデータが溜まったら配列に追加
  mediaRecorder.ondataavailable = (e) => {
    if (e.data && e.data.size > 0) {
      recordedChunks.push(e.data);
    }
  };

  // 録画停止時にダウンロード用リンクを生成
  mediaRecorder.onstop = () => {
    let fileExtension = '.mp4';
    let blob;
    if (MediaRecorder.isTypeSupported('video/mp4')) {
      blob = new Blob(recordedChunks, { type: 'video/mp4' });
      fileExtension = '.mp4';
    } else if (MediaRecorder.isTypeSupported('video/webm')) {
      blob = new Blob(recordedChunks, { type: 'video/webm' });
      fileExtension = '.webm';
    } else {
      console.error("no suitable mimetype found for this device");
    }
    const url = URL.createObjectURL(blob);

    downloadLink.href = url;
    downloadLink.style.display = 'inline-block';
    // サニタイズしたファイル名を利用
    const safeName = sanitizeFilename(fileInput.files[0].name.replace(/\.[^/.]+$/, ""));
    downloadLink.download = safeName + '_posed' + fileExtension;
    // JSONデータもダウンロードリンクに設定
    const jsonBlob = new Blob([JSON.stringify(poseJSON)], { type: 'application/json' });
    const jsonUrl = URL.createObjectURL(jsonBlob);
    downloadJSONLink.href = jsonUrl;
    downloadJSONLink.style.display = 'inline-block';
    downloadJSONLink.download = safeName + '_posed.json';
    // JSONデータ解析ページへのリンクを表示
    analyzeVideoJsonLink.style.display = 'inline-block';
  };

  // 録画スタート
  mediaRecorder.start();
  console.log('Recording started');

  // ボタン状態を更新
  startButton.disabled = true;
  stopButton.disabled = false;
  downloadLink.style.display = 'none';
});

/**
 * 録画停止ボタンが押された
 */
stopButton.addEventListener('click', () => {
  stopRecording();
});

/**
 * 録画停止処理
 */
function stopRecording() {
  if (mediaRecorder && mediaRecorder.state === 'recording') {
    mediaRecorder.stop();
    console.log('Recording stopped');
    startButton.disabled = false;
    stopButton.disabled = true;
  }
}