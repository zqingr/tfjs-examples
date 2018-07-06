// Trains a simple deep NN on the MNIST dataset.
// Gets to 98.40% test accuracy after 20 epochs
// (there is *a lot* of margin for parameter tuning).
// 17 seconds per epoch.

import * as tf from '@tensorflow/tfjs'
import { MnistData } from './data'
import { ChartBatchLog, ChartEpochLog } from '../../utils/charts'

const DROPOUT_RATE = 0.5

const model = tf.sequential()
model.add(tf.layers.dense({
  units: 512,
  activation: 'relu',
  inputShape: [784]
}))
model.add(tf.layers.dropout({ rate: DROPOUT_RATE }))
model.add(tf.layers.dense({
  units: 512,
  activation: 'relu'
}))
model.add(tf.layers.dropout({ rate: DROPOUT_RATE }))
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

async function train (data: MnistData) {
  const trainX = tf.tensor2d(data.trainImages, [55000, 784])
  const trainY = tf.tensor2d(data.trainLabels, [55000, 10])
  const testX = tf.tensor2d(data.testImages, [10000, 784])
  const testY = tf.tensor2d(data.testLabels, [10000, 10])

  console.group('data shape:')
  console.log('trainX.shape:', trainX.shape)
  console.log('trainY.shape:', trainY.shape)
  console.log('testX.shape:', testX.shape)
  console.log('testY.shape:', testY.shape)
  console.groupEnd()

  console.time('Totol training time')
  const history = await model.fit(trainX, trainY, {
    validationData: [testX, testY],
    batchSize: 128,
    epochs: 5,
    callbacks: {
      onBatchEnd: (epoch: number, log) => {
        console.log('batch:', epoch, log)
        if (epoch % 10 === 0) chartBatch.updata(log)
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
        chartEpoch.updata(log)
        return tf.nextFrame()
      }
    }
  })

  const score = model.evaluate(testX, testY)
  console.timeEnd('Totol training time')
  score[0].print()
  score[1].print()
}

async function load () {
  const data = new MnistData()
  await data.load()
  return data
}

async function start () {
  const data = await load()
  await train(data)
}

start()
