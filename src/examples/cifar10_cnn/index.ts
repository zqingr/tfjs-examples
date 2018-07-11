import * as tf from '@tensorflow/tfjs'
import { Cifar10 } from '../../datasets/cifar10'
import { ChartBatchLog, ChartEpochLog } from '../../utils/charts'

const BATCH_SIZE = 16
const TRAIN_BATCHES = 780
const TEST_BATCH_SIZE = 1000
const TEST_ITERATION_FREQUENCY = 5

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
      epochs: 50,
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

  console.log(history.history)
  // Call dispose on the training/test tensors to free their GPU memory.

  // tf.nextFrame() returns a promise that resolves at the next call to
  // requestAnimationFrame(). By awaiting this promise we keep our model
  // training from blocking the main UI thread and freezing the browser.
  await tf.nextFrame()

  // console.time('Totol training time')
  // const history = await model.fit(trainX, trainY, {
  //   validationData: [testX, testY],
  //   batchSize: 128,
  //   epochs: 5,
  //   callbacks: {
  //     onBatchEnd: (epoch: number, log) => {
  //       console.log('batch:', epoch, log)
  //       if (epoch % 10 === 0) chartBatch.updata(log)
  //       return tf.nextFrame()
  //     },
  //     onEpochBegin: (epoch, log) => {
  //       console.time('Epoch training time')
  //       console.groupCollapsed('epoch times:', epoch)
  //       return tf.nextFrame()
  //     },
  //     onEpochEnd: (epoch, log) => {
  //       console.groupEnd()
  //       console.timeEnd('Epoch training time')
  //       console.log(epoch, log)
  //       chartEpoch.updata(log)
  //       return tf.nextFrame()
  //     }
  //   }
  // })

  // const score = model.evaluate(testX, testY)
  // console.timeEnd('Totol training time')
  // score[0].print()
  // score[1].print()
}

async function load () {
  const data = new Cifar10()
  await data.load()
  await train(data)

  const {xs, ys} = data.nextTrainBatch(1500)
  console.log(xs, ys)
}

load()
