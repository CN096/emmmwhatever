# -*- coding: utf-8 -*-

from jpype import *

#路径
startJVM(getDefaultJVMPath(), "-Djava.class.path=D:\hanlp\hanlp-1.7.8.jar;D:\hanlp", "-Xms1g", "-Xmx1g")

#繁体转简体
def TraditionalChinese2SimplifiedChinese(sentence_str):
    HanLP = JClass('com.hankcs.hanlp.HanLP')
    return HanLP.convertToSimplifiedChinese(sentence_str)

#切词&命名实体识别与词性标注(可以粗略识别)
def NLP_tokenizer(sentence_str):
    NLPTokenizer = JClass('com.hankcs.hanlp.tokenizer.NLPTokenizer')
    return NLPTokenizer.segment(sentence_str)

#地名识别，标注为ns
def Place_Recognize(sentence_str):
    HanLP = JClass('com.hankcs.hanlp.HanLP')
    segment = HanLP.newSegment().enablePlaceRecognize(True)
    return HanLP.segment(sentence_str)

#人名识别,标注为nr
def PersonName_Recognize(sentence_str):
    HanLP = JClass('com.hankcs.hanlp.HanLP')
    segment = HanLP.newSegment().enableNameRecognize(True)
    return HanLP.segment(sentence_str)

#机构名识别,标注为nt
def Organization_Recognize(sentence_str):
    HanLP = JClass('com.hankcs.hanlp.HanLP')
    segment = HanLP.newSegment().enableOrganizationRecognize(True)
    return HanLP.segment(sentence_str)

#标注结果转化成列表
def total_result(function_result_input):
    x = str(function_result_input)
    y = x[1:len(x)-1]
    y = y.split(',')
    return y

#时间实体
def time_result(total_result):
    z = []
    for i in range(len(total_result)):
        if total_result[i][-2:] == '/t':
            z.append(total_result[i])
    return z

#Type_Recognition 可以选 ‘place’,‘person’,‘organization’三种实体,
#返回单一实体类别的列表
def single_result(Type_Recognition,total_result):
    if Type_Recognition == 'place':
        Type = '/ns'
    elif Type_Recognition == 'person':
        Type = '/nr'
    elif Type_Recognition == 'organization':
        Type = '/nt'
    else:
        print ('请输入正确的参数：（place，person或organization）')
    z = []
    for i in range(len(total_result)):
        if total_result[i][-3:] == Type:
            z.append(total_result[i])
    return z

#把单一实体结果汇总成一个字典
def dict_result(sentence_str):
    sentence = TraditionalChinese2SimplifiedChinese(sentence_str)
    total_dict = {}
    a = total_result(Place_Recognize(sentence))
    b = single_result('place',a)
    c = total_result(PersonName_Recognize(sentence))
    d = single_result('person',c)
    e = total_result(Organization_Recognize(sentence))
    f = single_result('organization',e)
    g = total_result(NLP_tokenizer(sentence))
    h = time_result(g)
    total_list = [i for i in [b,d,f,h]]
    total_dict.update(place = total_list[0],person = total_list[1],organization = total_list[2],time = total_list[3])
    shutdownJVM()#关闭JVM虚拟机
    return total_dict

#测试
test_sentence="2018年武胜县新学乡政府大楼门前锣鼓喧天,6月份蓝翔给宁夏固原市彭阳县红河镇捐赠了挖掘机,中国科学院计算技术研究所的宗成庆教授负责教授自然语言处理课程"
print (dict_result(test_sentence))