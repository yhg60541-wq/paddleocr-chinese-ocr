import Taro from '@tarojs/taro'
import type { OcrResult, OcrPrediction } from '@/types/ocr'

// Flask 后端地址 — 本地 CPU 推理
// 说明: 微信开发者工具模拟器里 localhost 指向沙箱，需改成电脑的局域网 IP
//       Flask 启动日志会打印 "Running on http://xxx.xxx.xxx.xxx:5000"
//       真机调试时同样使用局域网 IP
const API_BASE = 'http://127.0.0.1:5000'

/**
 * OCR 文字识别
 * @param filePath 图片本地路径
 * @returns OcrResult
 */
export async function recognizeImage(filePath: string): Promise<OcrResult> {
  console.info('[OCR] 开始识别:', filePath)
  
  // 读文件 → base64
  const fs = Taro.getFileSystemManager()
  const base64 = fs.readFileSync(filePath, 'base64')
  
  const res = await Taro.request({
    url: `${API_BASE}/ocr`,
    method: 'POST',
    data: { image: base64 },
    header: { 'Content-Type': 'application/json' },
    timeout: 30000
  })
  
  if (res.statusCode !== 200) {
    console.error('[OCR] 请求失败:', res.statusCode, res.data)
    throw new Error(`识别失败: HTTP ${res.statusCode}`)
  }
  
  if (res.data?.error) {
    throw new Error(res.data.error)
  }
  
  return res.data as OcrResult
}

/**
 * 健康检查
 */
export async function checkHealth(): Promise<boolean> {
  try {
    const res = await Taro.request({
      url: `${API_BASE}/health`,
      method: 'GET',
      timeout: 3000
    })
    return res.data?.status === 'ok'
  } catch {
    return false
  }
}

/**
 * 获取模型信息
 */
export async function getModelInfo(): Promise<any> {
  try {
    const res = await Taro.request({
      url: `${API_BASE}/model_info`,
      method: 'GET',
      timeout: 3000
    })
    return res.data
  } catch {
    return null
  }
}

/**
 * 选择图片 (拍照或从相册)
 */
export function chooseImage(sourceType: ['camera'] | ['album'] | ['camera', 'album'] = ['camera', 'album']) {
  return Taro.chooseImage({
    count: 1,
    sourceType,
    sizeType: ['compressed']
  })
}

/**
 * 保存历史记录到本地
 */
export function saveHistory(item: { imagePath: string; thumbnail: string; predictions: OcrPrediction[] }) {
  const key = 'ocr_history'
  const history = getHistory()
  history.unshift({
    id: Date.now().toString(),
    ...item,
    timestamp: Date.now()
  })
  // 最多保留 50 条
  if (history.length > 50) history.length = 50
  Taro.setStorageSync(key, history)
}

export function getHistory(): any[] {
  return Taro.getStorageSync('ocr_history') || []
}

export function clearHistory() {
  Taro.removeStorageSync('ocr_history')
}
