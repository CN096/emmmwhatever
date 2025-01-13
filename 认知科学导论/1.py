# -*- coding:utf-8 -*-
from pyhanlp import *

content = "现如今，机器学习和深度学习带动人工智能飞速的发展，并在图片处理、语音识别领域取得巨大成功。"
print(HanLP.segment(content))

content = "马伊琍与文章宣布离婚，华为是背后的赢家。"
print('原句:' + content)
print(HanLP.segment(content))

# 添加自定义词典
# insert会覆盖字典中已经存在的词，add会跳过已经存在的词，
# add("文章"，"nr 300") ,nr为词性，300为词频； add("交易平台","nz 1024 n 1") 表示可以一词多性 ，交易平台词性即可为nz 词频为1024，也可为n 词频为1
CustomDictionary.add("文章", "nr 300")
CustomDictionary.insert("工程机械", "nz 1024")
CustomDictionary.add("交易平台", "nz 1024 n 1")
print(HanLP.segment(content))

segment = HanLP.newSegment().enableNameRecognize(True)
print(HanLP.segment(content))
