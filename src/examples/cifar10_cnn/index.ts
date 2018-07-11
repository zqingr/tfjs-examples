// Trains a simple deep NN on the MNIST dataset.
// Gets to 75% test accuracy after 6 epochs
// (there is *a lot* of margin for parameter tuning).
// 110 seconds per epoch.

import * as tf from '@tensorflow/tfjs'
import { Cifar10 } from '../../datasets/cifar10'
import { ChartBatchLog, ChartEpochLog } from '../../utils/charts'

const BATCH_SIZE = 32

const model = tf.sequential()
model.add(tf.layers.conv2d({
  kernelSize: 3,
  filters: 32,
  activation: 'relu',
  padding: 'same',
  inputShape: [32, 32, 3]
}))
model.add(tf.layers.conv2d({
  kernelSize: 3,
  filters: 32,
  activation: 'relu'
}))
model.add(tf.layers.maxPooling2d({poolSize: [2, 2]}))
model.add(tf.layers.dropout({ rate: 0.25 }))

model.add(tf.layers.conv2d({
  kernelSize: 3,
  filters: 64,
  activation: 'relu',
  padding: 'same'
}))
model.add(tf.layers.conv2d({
  kernelSize: 3,
  filters: 64,
  activation: 'relu'
}))
model.add(tf.layers.maxPooling2d({poolSize: [2, 2]}))
model.add(tf.layers.dropout({ rate: 0.25 }))

model.add(tf.layers.flatten())
model.add(tf.layers.dense({
  units: 512,
  activation: 'relu'
}))
model.add(tf.layers.dropout({ rate: 0.5 }))
model.add(tf.layers.dense({
  units: 10,
  activation: 'softmax'
}))

model.summary()

const chartBatch = new ChartBatchLog().init('mountNode')
const chartEpoch = new ChartEpochLog().init('mountNode1')

const optimizer = tf.train.adam()
model.compile({
  optimizer,
  loss: 'categoricalCrossentropy',
  metrics: ['accuracy']
})

async function train (data: Cifar10) {
  // The entire dataset doesn't fit into memory so we call train repeatedly
  // with batches using the fit() method.
  const { xs: x, ys: y } = data.nextTrainBatch()
  const history = await model.fit(
    x.reshape([50000, 32, 32, 3]), y, {
      batchSize: BATCH_SIZE,
      epochs: 6,
      callbacks: {
        onBatchEnd: (epoch: number, log) => {
          console.log('batch:', epoch, log)
          if (epoch % 100 === 0) chartBatch.updata(log)
          return tf.nextFrame()
        },
        onEpochBegin: (epoch, log) => {
          console.time('Epoch training time')
          console.groupCollapsed('epoch times:', epoch)
          return tf.nextFrame()
        },
        onEpochEnd: (epoch, log) => {
          console.groupEnd()
          console.timeEnd('Epoch training time')
          console.log(epoch, log)

          const testBatch = data.nextTestBatch(2000)
          const score = model.evaluate(testBatch.xs.reshape([2000, 32, 32, 3]), testBatch.ys)
          console.timeEnd('Totol training time')
          score[0].print()
          score[1].print()

          chartEpoch.updata(log)
          return tf.nextFrame()
        }
      }
    })

  await tf.nextFrame()
}

async function load () {
  const data = new Cifar10()
  await data.load()
  await train(data)

  const {xs, ys} = data.nextTrainBatch(1500)
  console.log(xs, ys)
}

load()
