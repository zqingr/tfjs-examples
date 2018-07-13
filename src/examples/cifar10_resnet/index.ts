import * as tf from '@tensorflow/tfjs'
import { Cifar10 } from '../../datasets/cifar10'
import { ChartBatchLog, ChartEpochLog } from '../../utils/charts'

function resnetBlock (x: tf.SymbolicTensor, f: number, [f1, f2, f3]: [number, number, number], stage: number, block: string, s?: number) {
  // defining name basis
  const convNameBase = 'res' + stage + block + '_branch'
  const bnNameBase = 'bn' + stage + block + '_branch'

  // First component of main path
  let newX = tf.layers.conv2d({ filters: f1, kernelSize: [1, 1], strides: [s || 1, s || 1], padding: 'valid', name: convNameBase + '2a', kernelInitializer: tf.initializers.glorotUniform({ seed: 0 }) }).apply(x) as tf.SymbolicTensor
  newX = tf.layers.batchNormalization({axis: 3, name: bnNameBase + '2a'}).apply(newX) as tf.SymbolicTensor
  newX = tf.layers.activation({activation: 'relu'}).apply(newX) as tf.SymbolicTensor

  // Second component of main path
  newX = tf.layers.conv2d({ filters: f2, kernelSize: [f, f], strides: [1, 1], padding: 'same', name: convNameBase + '2b', kernelInitializer: tf.initializers.glorotUniform({ seed: 0 }) }).apply(newX) as tf.SymbolicTensor
  newX = tf.layers.batchNormalization({axis: 3, name: bnNameBase + '2b'}).apply(newX) as tf.SymbolicTensor
  newX = tf.layers.activation({activation: 'relu'}).apply(newX) as tf.SymbolicTensor

  // Third component of main path
  newX = tf.layers.conv2d({ filters: f3, kernelSize: [1, 1], strides: [1, 1], padding: 'valid', name: convNameBase + '2c', kernelInitializer: tf.initializers.glorotUniform({ seed: 0 }) }).apply(newX) as tf.SymbolicTensor
  newX = tf.layers.batchNormalization({axis: 3, name: bnNameBase + '2c'}).apply(newX) as tf.SymbolicTensor

  let xShortcut: tf.SymbolicTensor | null = null
  if (s) {
    xShortcut = tf.layers.conv2d({filters: f3, kernelSize: [1, 1], strides: [s, s], padding: 'valid', name: convNameBase + '1', kernelInitializer: tf.initializers.glorotUniform({ seed: 0 })}).apply(x) as tf.SymbolicTensor
    xShortcut = tf.layers.batchNormalization({axis: 3, name: bnNameBase + '1'}).apply(xShortcut) as tf.SymbolicTensor
  }

  newX = tf.layers.add().apply([newX, xShortcut || x]) as tf.SymbolicTensor
  newX = tf.layers.activation({activation: 'relu'}).apply(newX) as tf.SymbolicTensor

  return newX
}

function resNet50 (inputShape: number[], classess: number) {
  // Define the input as a tensor with shape inputShape
  const xInput = tf.input({ shape: inputShape })

  // Zero padding
  let x = tf.layers.zeroPadding2d({ padding: [3, 3], inputShape }).apply(xInput) as tf.SymbolicTensor

  // stage 1
  x = tf.layers.conv2d({ filters: 64, kernelSize: [7, 7], strides: [2, 2], name: 'conv1', kernelInitializer: tf.initializers.glorotUniform({ seed: 0 }) }).apply(x) as tf.SymbolicTensor
  x = tf.layers.batchNormalization({ axis: 3, name: 'bn_conv1' }).apply(x) as tf.SymbolicTensor
  x = tf.layers.activation({ activation: 'relu' }).apply(x) as tf.SymbolicTensor
  x = tf.layers.maxPooling2d({ poolSize: [3, 3], strides: [2, 2] }).apply(x) as tf.SymbolicTensor

  // stage 2
  x = resnetBlock(x, 3, [64, 64, 256], 2, 'a', 1)
  x = resnetBlock(x, 3, [64, 64, 256], 2, 'b')
  x = resnetBlock(x, 3, [64, 64, 256], 2, 'c')

  // stage 3
  x = resnetBlock(x, 3, [128, 128, 512], 3, 'a', 2)
  x = resnetBlock(x, 3, [128, 128, 512], 3, 'b')
  x = resnetBlock(x, 3, [128, 128, 512], 3, 'c')
  x = resnetBlock(x, 3, [128, 128, 512], 3, 'd')

  // stage 4
  x = resnetBlock(x, 3, [256, 256, 1024], 4, 'a', 2)
  x = resnetBlock(x, 3, [256, 256, 1024], 4, 'b')
  x = resnetBlock(x, 3, [256, 256, 1024], 4, 'c')
  x = resnetBlock(x, 3, [256, 256, 1024], 4, 'd')
  x = resnetBlock(x, 3, [256, 256, 1024], 4, 'e')
  x = resnetBlock(x, 3, [256, 256, 1024], 4, 'f')

  // stage 5
  x = resnetBlock(x, 3, [512, 512, 2048], 5, 'a', 2)
  x = resnetBlock(x, 3, [512, 512, 2048], 5, 'b')
  x = resnetBlock(x, 3, [512, 512, 2048], 5, 'c')

  x = tf.layers.averagePooling2d({ poolSize: [2, 2], padding: 'same' }).apply(x) as tf.SymbolicTensor

  x = tf.layers.flatten().apply(x) as tf.SymbolicTensor
  x = tf.layers.dense({ units: classess, activation: 'softmax', name: 'fc' + classess, kernelInitializer: tf.initializers.glorotUniform({seed: 0}) }).apply(x) as tf.SymbolicTensor

  return tf.model({ inputs: xInput, outputs: x, name: 'ResNet50' })
}

// const x = tf.randomNormal([3, 4, 4, 6])
// const A = resnetBlock(x, 2, [2, 4, 6], 1, 'a', 2)

const model = resNet50([32, 32, 3], 10)

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
    x.reshape([50000, 32, 32, 3]), y, {
      batchSize: 32,
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
