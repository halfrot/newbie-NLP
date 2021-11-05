import arpa

models = arpa.loadf("train.lm", encoding="UTF-8")
print(models)

lm = models[0]
print(lm.vocabulary())
print(lm.counts())
print(lm.order())

print(lm.p("我们 的 勇气"))
print(lm.p("我 们的 勇气"))
print(lm.log_p("我们 的勇 气"))
print(lm.log_p("我 们的 勇气"))

print("sentences")
print(lm.s("评估 时 采用 了 不同 的 方法"))
print(lm.log_s("然后 ， 他 死 了"))
print(lm.log_s("今天 早上 ， 提交 一个 报告"))
print(lm.log_s("往往 总 是 搭配 在 一起"))
print(lm.log_s("中国 吃饭 篮球 旅游 ， 然后 了 也许"))

print("unknown")
print(lm.log_s("srilm 是 非常 强大 的 工具 ， 可 供 免费 使用"))
print(lm.log_s("<unk> 是 非常 强大 的 工具 ， 可 供 免费 使用"))
