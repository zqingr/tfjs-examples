import * as tf from '@tensorflow/tfjs'
import { Cifar10 } from '../../datasets/cifar10'
import { ChartBatchLog, ChartEpochLog } from '../../utils/charts'

async function load () {
  const data = new Cifar10()
  await data.load()
  console.log(data)
}

load()
