import { useState, useEffect } from 'react'
import { View, Text, Image } from '@tarojs/components'
import Taro from '@tarojs/taro'
import styles from './index.module.scss'
import { getHistory, clearHistory } from '@/services/ocr'

interface HistoryPageProps {}

export default function HistoryPage() {
  const [list, setList] = useState<any[]>([])

  useEffect(() => {
    const data = getHistory()
    setList(data)
    console.log('[History] 加载', data.length, '条记录')
  }, [])

  const handleClear = () => {
    Taro.showModal({
      title: '清空历史',
      content: '确定清空所有识别历史？',
      success: (res) => {
        if (res.confirm) {
          clearHistory()
          setList([])
          Taro.showToast({ title: '已清空', icon: 'success' })
        }
      }
    })
  }

  const formatTime = (ts: number) => {
    const d = new Date(ts)
    const pad = (n: number) => String(n).padStart(2, '0')
    return `${d.getMonth() + 1}/${d.getDate()} ${pad(d.getHours())}:${pad(d.getMinutes())}`
  }

  return (
    <View className={styles.historyPage}>
      {list.length === 0 ? (
        <View className={styles.empty}>
          <Text className={styles.emptyIcon}>📭</Text>
          <Text className={styles.emptyText}>暂无识别历史</Text>
        </View>
      ) : (
        <>
          <View className={styles.historyList}>
            {list.map((item) => (
              <View key={item.id} className={styles.historyCard}>
                <View className={styles.historyHeader}>
                  {item.thumbnail ? (
                    <Image className={styles.historyThumb} src={item.thumbnail} mode='aspectFill' />
                  ) : (
                    <View className={styles.historyThumb} />
                  )}
                  <View className={styles.historyMeta}>
                    <Text className={styles.historyText}>
                      {(item.predictions || []).map((p: any) => p.transcription).join(' | ') || '(无文字)'}
                    </Text>
                    <Text className={styles.historyTime}>{formatTime(item.timestamp)}</Text>
                  </View>
                  <Text className={styles.countBadge}>{item.predictions?.length || 0}条</Text>
                </View>
              </View>
            ))}
          </View>
          <Text className={styles.clearBtn} onClick={handleClear}>
            清空历史
          </Text>
        </>
      )}
    </View>
  )
}
