# -*- coding: utf-8 -*-

from pyspark import SparkConf
from pyspark.sql import SparkSession
import sys, traceback
import uuid

def process(spark, calc_date):
    sql = """ 
        CREATE EXTERNAL TABLE IF NOT EXISTS app.app_shw_test(
            id bigint COMMENT 'ID', 
            name string COMMENT '名称'
        )
        COMMENT '测试表'
        PARTITIONED BY (dt string)
        STORED AS ORC
        LOCATION '/user/shw/app_shw_test'
        tblproperties ('orc.compress'='SNAPPY')
    """
    spark.sql(sql)
    sql = """ select id, name, '{calc_date}' dtfrom xx_testwhere dt = '{calc_date}'group by id,name """.format(
        calc_date=calc_date)
    df = spark.sql(sql)
    df.write.mode("append").insertInto('app_shw_test', overwrite=True)

def createSparkSession(appName):
    conf = SparkConf().setAppName(appName)
    conf.set("spark.rdd.compress", "true")
    conf.set("spark.broadcast.compress", "true")
    conf.set("hive.exec.dynamic.partition", "true")
    conf.set("hive.exec.dynamic.partition.mode", "nonstrict")
    conf.set("hive.exec.max.dynamic.partitions", "100000")
    conf.set("hive.exec.max.dynamic.partitions.pernode", "100000")
    conf.set("hive.auto.convert.join", "true")
    conf.set("mapred.max.split.size", "256000000")  # 每个Map最大输入大小
    conf.set("mapred.min.split.size.per.node", "100000000")  # 一个节点上split的至少的大小
    conf.set("mapred.min.split.size.per.rack", "100000000")  # 一个交换机下split的至少的大小
    conf.set("hive.input.format", "org.apache.hadoop.hive.ql.io.CombineHiveInputFormat")  # 执行Map前进行小文件合并
    conf.set("hive.merge.mapfiles", "true")  # 在Map-only的任务结束时合并小文件
    conf.set("hive.merge.mapredfiles", "true")  # 在Map-Reduce的任务结束时合并小文件
    conf.set("hive.merge.size.per.task", "256*1000*1000")  # 合并文件的大小
    conf.set("hive.merge.smallfiles.avgsize", "16000000")  # 当输出文件的平均大小小于该值时，启动一个独立的map-reduce任务进行文件merge
    conf.set("spark.sql.shuffle.partitions", "500")  # 设置shuffle分区数
    conf.set("spark.driver.maxResultSize", "5g")
    spark = SparkSession.builder.config(conf=conf).enableHiveSupport().getOrCreate()
    return spark

def main():
    calc_date = sys.argv[1]
    appName = "spark_hive_task-" + str(uuid.uuid1())
    spark = createSparkSession(appName=appName)
    try:
        process(spark, calc_date)
    except Exception as ex:
        traceback.print_exc()
    finally:
        spark.stop()

if __name__ == "__main__":
    main()