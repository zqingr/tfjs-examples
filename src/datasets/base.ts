import * as tf from '@tensorflow/tfjs'

export class DataSet {
  IMG_WIDTH: number
  IMG_HEIGHT: number
  TRAIN_IMAGES: string[]
  TRAIN_LABLES: string
  TEST_IMAGES: string[]
  TEST_LABLES: string

  trainDatas: tf.Tensor<tf.Rank.R4>[]
  testDatas: tf.Tensor<tf.Rank.R4>[]
  trainLables: string[][]
  testLables: string[][]

  trainM: number = 0
  testM: number = 0

  loadImg (src: string): Promise<HTMLImageElement> {
    return new Promise((resolve, reject) => {
      const img = new Image()
      img.src = src
      img.onload = () => { resolve(img) }
      img.onerror = reject
    })
  }

  loadImages (srcs: string[]): Promise<tf.Tensor4D[]> {
    return Promise.all(srcs.map(this.loadImg)).then(async imgs => imgs
      .map(img => tf.fromPixels(img).reshape([img.naturalHeight, this.IMG_WIDTH, this.IMG_HEIGHT, 3]) as tf.Tensor4D))
  }

  async load () {
    this.trainDatas = await this.loadImages(this.TRAIN_IMAGES)
    this.testDatas = await this.loadImages(this.TEST_IMAGES)

    this.trainLables = await fetch(this.TRAIN_LABLES).then(res => res.json())
    this.testLables = await fetch(this.TEST_LABLES).then(res => res.json())

    this.trainLables.forEach(lable => { this.trainM += lable.length })
    this.testLables.forEach(lable => { this.testM += lable.length })
  }
}
