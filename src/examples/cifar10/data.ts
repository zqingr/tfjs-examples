import * as tf from '@tensorflow/tfjs'

const TRAIN_IMAGES = [
  '/datasets/cifar10/data_batch_1.png',
  '/datasets/cifar10/data_batch_2.png',
  '/datasets/cifar10/data_batch_3.png',
  '/datasets/cifar10/data_batch_4.png',
  '/datasets/cifar10/data_batch_5.png'
]
const TRAIN_LABLES = '/datasets/cifar10/train_lables.json'
const TEST_IMAGES = [
  '/datasets/cifar10/test_batch.png'
]
const TEST_LABLES = '/datasets/cifar10/test_lables.json'
const IMG_WIDTH = 32
const IMG_HEIGHT = 32

export class DataSet {
  imgWidth: number
  imgHeight: number
  trainImagesSrc: string[]
  trainLablesJson: string
  testImagesSrc: string[]
  testLablesJson: string

  trainDatas: tf.Tensor<tf.Rank.R4>[]
  testDatas: tf.Tensor<tf.Rank.R4>[]
  trainLables: string[][]
  testLables: string[][]

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
      .map(img => tf.fromPixels(img).reshape([img.naturalHeight, this.imgWidth, this.imgHeight, 3]) as tf.Tensor4D))
  }

  async load () {
    this.trainDatas = await this.loadImages(TRAIN_IMAGES)
    this.testDatas = await this.loadImages(TEST_IMAGES)

    this.trainLables = await fetch(TRAIN_LABLES).then(res => res.json())
    this.testLables = await fetch(TEST_LABLES).then(res => res.json())
  }
}
