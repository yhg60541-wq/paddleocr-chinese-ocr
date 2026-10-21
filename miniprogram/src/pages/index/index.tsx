import { useState, useEffect } from 'react'
import { View, Text, Image, Button } from '@tarojs/components'
import Taro from '@tarojs/taro'
import classnames from 'classnames'
import styles from './index.module.scss'
import { chooseImage, recognizeImage, saveHistory, checkHealth } from '@/services/ocr'
import type { OcrPrediction } from '@/types/ocr'

interface OcrPageProps {}

export default function OcrPage() {
  const [imagePath, setImagePath] = useState<string>('')
  const [annotatedPath, setAnnotatedPath] = useState<string>('')
  const [predictions, setPredictions] = useState<OcrPrediction[]>([])
  const [loading, setLoading] = useState(false)
  const [serverOk, setServerOk] = useState<boolean | null>(null)

  // 启动时检查服务
  useEffect(() => {
    checkHealth().then(setServerOk)
  }, [])

  const handleChoose = async (sourceType: ['camera'] | ['album'] | ['camera', 'album']) => {
    try {
      const res = await chooseImage(sourceType)
      if (!res.tempFilePaths?.length) return
      const path = res.tempFilePaths[0]
      setImagePath(path)
      setAnnotatedPath('')
      setPredictions([])
    } catch (e) {
      console.error('[OCR] 选图失败:', e)
    }
  }

  const handleRecognize = async () => {
    if (!imagePath) {
      Taro.showToast({ title: '请先选图', icon: 'none' })
      return
    }
    setLoading(true)
    try {
      const result = await recognizeImage(imagePath)
      setPredictions(result.detections || [])

      // 标注图存本地
      if (result.annotated_image) {
        const base64 = result.annotated_image
        const fs = Taro.getFileSystemManager()
        const tmpPath = `${Taro.env.USER_DATA_PATH}/annotated_${Date.now()}.jpg`
        fs.writeFileSync(tmpPath, base64, 'base64')
        setAnnotatedPath(tmpPath)
      }

      // 保存历史
      saveHistory({
        imagePath,
        thumbnail: '',
        predictions: result.detections || []
      })

      Taro.showToast({ title: '识别完成', icon: 'success' })
    } catch (e: any) {
      console.error('[OCR] 识别失败:', e)
      Taro.showToast({ title: e?.message || '识别失败', icon: 'none', duration: 3000 })
      setServerOk(false)
    } finally {
      setLoading(false)
    }
  }

  const handleClear = () => {
    setImagePath('')
    setAnnotatedPath('')
    setPredictions([])
  }

  const handleCopy = (text: string) => {
    Taro.setClipboardData({
      data: text,
      success: () => Taro.showToast({ title: '已复制', icon: 'success' })
    })
  }

  return (
    <View className={styles.ocrPage}>
      {/* Header */}
      <View className={styles.header}>
        <Text className={styles.title}>OCR 文字识别</Text>
        <Text className={styles.subtitle}>拍照或选图，自动检测并识别文字</Text>
      </View>

      {/* Server Status */}
      <View className={styles.serverStatus}>
        <Text className={serverOk ? styles.statusOk : serverOk === false ? styles.statusErr : ''}>
          {serverOk === null ? '检查服务...' : serverOk ? '● 推理服务正常' : '● 推理服务不可用'}
        </Text>
      </View>

      {/* Image Card */}
      <View className={styles.imageCard}>
        <View className={styles.imageWrapper}>
          {annotatedPath || imagePath ? (
            <Image
              className={styles.imagePreview}
              src={annotatedPath || imagePath}
              mode='aspectFit'
            />
          ) : (
            <View className={styles.placeholder}>
              <Text className={styles.placeholderIcon}>📷</Text>
              <Text className={styles.placeholderText}>点击下方按钮拍照或选图</Text>
            </View>
          )}
          {loading && (
            <View className={styles.loading}>
              <Text>识别中...</Text>
              <Text className={styles.loadingText}>AI 正在分析文字</Text>
            </View>
          )}
        </View>

        {/* Action Buttons */}
        <View className={styles.actionBar}>
          <Button
            className={classnames(styles.btn, styles.btnSecondary)}
            onClick={() => handleChoose(['album'])}
          >
            从相册选
          </Button>
          <Button
            className={classnames(styles.btn, styles.btnPrimary)}
            onClick={handleRecognize}
            disabled={!imagePath || loading}
          >
            {loading ? '识别中...' : '开始识别'}
          </Button>
        </View>
      </View>

      {/* Result Card */}
      <View className={styles.resultCard}>
        <View className={styles.resultHeader}>
          <Text className={styles.resultTitle}>识别结果</Text>
          <Text className={styles.resultCount}>
            {predictions.length > 0 ? `${predictions.length} 条文字` : ''}
          </Text>
        </View>

        {predictions.length === 0 ? (
          <View className={styles.resultEmpty}>
            {imagePath ? '未检测到文字' : '等待识别...'}
          </View>
        ) : (
          <View className={styles.textList}>
            {predictions.map((p, i) => (
              <View
                key={i}
                className={styles.textItem}
                onClick={() => handleCopy(p.transcription)}
              >
                <Text className={styles.textIndex}>{i + 1}</Text>
                <Text className={styles.textContent}>{p.transcription || '(空)'}</Text>
                <Text className={styles.textConf}>{(p.confidence * 100).toFixed(0)}%</Text>
              </View>
            ))}
          </View>
        )}
      </View>

      {/* Clear button */}
      {(imagePath || predictions.length > 0) && (
        <View style={{ marginTop: '32rpx', textAlign: 'center' }}>
          <Text
            style={{ color: '#86909c', fontSize: '24rpx' }}
            onClick={handleClear}
          >
            清除重新识别
          </Text>
        </View>
      )}
    </View>
  )
}
