import { DataSet } from '../base'

export class Cifar10 extends DataSet {
  TRAIN_IMAGES = [
    '/datasets/cifar10/data_batch_1.png',
    '/datasets/cifar10/data_batch_2.png',
    '/datasets/cifar10/data_batch_3.png',
    '/datasets/cifar10/data_batch_4.png',
    '/datasets/cifar10/data_batch_5.png'
  ]
  TRAIN_LABLES = '/datasets/cifar10/train_lables.json'
  TEST_IMAGES = [
    '/datasets/cifar10/test_batch.png'
  ]
  TEST_LABLES = '/datasets/cifar10/test_lables.json'
  IMG_WIDTH = 32
  IMG_HEIGHT = 32
  NUM_CLASSES = 10
}
