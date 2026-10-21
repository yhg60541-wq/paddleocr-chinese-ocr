export interface OcrPrediction {
  points: number[]          // 8 个数 [x1,y1,x2,y2,x3,y3,x4,y4]
  transcription: string     // 识别文字
  confidence: number        // 0~1
}

export interface OcrResult {
  detections: OcrPrediction[]
  annotated_image?: string  // base64
  count: number
}

export interface HistoryItem {
  id: string
  imagePath: string        // 本地临时路径
  thumbnail: string        // 缩略图 base64
  predictions: OcrPrediction[]
  timestamp: number
}
