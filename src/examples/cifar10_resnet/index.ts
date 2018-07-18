import * as tf from '@tensorflow/tfjs'
import { Cifar10 } from 'tfjs-cifar10-web'
import { ChartBatchLog, ChartEpochLog } from '../../utils/charts'

import { resnetV2 } from 'tfjs-applications'

// const x = tf.randomNormal([3, 4, 4, 6])
// const A = resnetBlock(x, 2, [2, 4, 6], 1, 'a', 2)

const model = resnetV2({ inputShape: [32, 32, 3], depth: 11 })

const optimizer = tf.train.adam()
model.compile({
  optimizer,
  loss: 'categoricalCrossentropy',
  metrics: ['accuracy']
})
model.summary()

const chartBatch = new ChartBatchLog().init('mountNode')
const chartEpoch = new ChartEpochLog().init('mountNode1')

async function train (data: Cifar10) {
  // The entire dataset doesn't fit into memory so we call train repeatedly
  // with batches using the fit() method.
  const { xs: x, ys: y } = data.nextTrainBatch()
  const history = await model.fit(
    x.reshape([50000, 32, 32, 3]) as any, y as any, {
      batchSize: 32,
      epochs: 200,
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
          const score = model.evaluate(testBatch.xs.reshape([2000, 32, 32, 3]) as any, testBatch.ys as any)
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
