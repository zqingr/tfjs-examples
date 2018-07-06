import echarts, { ECharts } from 'echarts'

class ChartLog {
  legendData: string[]
  series: any[]
  data: any[] = []
  chart: ECharts
  title: string = ''

  currentIndex: number = 0

  init (mountName: string) {
    const chart = echarts.init(document.getElementById(mountName) as HTMLDivElement)
    chart.setOption({
      title: {
        text: this.title
      },
      tooltip: {
        trigger: 'axis'
      },
      legend: {
        data: this.legendData
      },
      grid: {
        left: '3%',
        right: '4%',
        bottom: '3%',
        containLabel: true
      },
      toolbox: {
        feature: {
          saveAsImage: {}
        }
      },
      xAxis: {
        type: 'category',
        boundaryGap: false,
        data: []
      },
      yAxis: {
        type: 'value'
      },
      series: this.series
    })
    this.chart = chart
    return this
  }

  getData (): any[] { return [] }

  updata (log: any) {
    this.data.push(log)
    this.chart.setOption({
      xAxis: {
        data: Array(++this.currentIndex).fill(0).map((n, i) => i)
      },
      series: this.getData()
    })
  }
}

export class ChartEpochLog extends ChartLog {
  title = 'Epoch Log'
  legendData = ['loss', 'acc', 'val_acc', 'val_loss']
  series = [
    {
      name: 'loss',
      type: 'line',
      stack: 'loss',
      data: []
    },
    {
      name: 'acc',
      type: 'line',
      stack: 'acc',
      data: []
    },
    {
      name: 'val_acc',
      type: 'line',
      stack: 'val_acc',
      data: []
    },
    {
      name: 'val_loss',
      type: 'line',
      stack: 'val_loss',
      data: []
    }
  ]

  getData () {
    return [{
      name: 'loss',
      data: this.data.map(log => log.loss)
    }, {
      name: 'acc',
      data: this.data.map(log => log.acc)
    }, {
      name: 'val_acc',
      data: this.data.map(log => log.val_acc)
    }, {
      name: 'val_loss',
      data: this.data.map(log => log.val_loss)
    }]
  }
}

export class ChartBatchLog extends ChartLog {
  legendData = ['loss', 'acc']
  title = 'Batch Log'
  series = [
    {
      name: 'loss',
      type: 'line',
      stack: 'loss',
      data: []
    },
    {
      name: 'acc',
      type: 'line',
      stack: 'acc',
      data: []
    }
  ]
  getData () {
    return [{
      name: 'loss',
      data: this.data.map(log => log.loss)
    }, {
      name: 'acc',
      data: this.data.map(log => log.acc)
    }]
  }
}
