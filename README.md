# SnapMath

# Метрики. Методы оценки моделей классификации.

# Метод F1

F1 полезен в условиях, когда распределение результатов неравномерно или когда стоимость ложноположительных и ложноотрицательных результатов значительна.  
F1 — это среднее гармоническое между точностью (Precision) и полнотой (Recall):

![Formula F1](./img/ml-score/image1.png)

### Precision (Точность)

Precision — это доля правильных положительных предсказаний среди всех предсказанных положительных:

![Formula F1](./img/ml-score/image2.png)

Где:
- **TP** — количество истинно положительных предсказаний (True Positives).
- **FP** — количество ложноположительных предсказаний (False Positives).

### Recall (Полнота)

Recall — это доля правильных положительных предсказаний среди всех реальных положительных:

![Formula F1](./img/ml-score/image3.png)

Где:
- **FN** — количество ложных отрицательных предсказаний (False Negatives).

### Неравномерное распределение результатов

Когда мы говорим, что распределение результатов неравномерное, мы имеем в виду ситуацию, когда классы в данных сильно не сбалансированы.  
Например, в задаче классификации email'ов на «спам» и «не спам» большинство писем могут быть «не спамом», а совсем немного — «спамом».

Если использовать просто точность (Accuracy), то модель, которая всегда предсказывает "не спам", будет иметь высокую точность, так как подавляющее большинство писем действительно не спам. Однако такая модель будет не полезной, так как она не сможет правильно классифицировать спам.

В таких случаях F1-метрика дает более информативную оценку, потому что она балансирует точность и полноту, предотвращая смещение в сторону самого распространенного класса и учитывая ошибки в обоих направлениях:
- **Ложноположительные (FP):** когда модель предсказывает положительный класс, но на самом деле это отрицательный класс.  
  Например, когда система определяет нормальное письмо как спам.
- **Ложноотрицательные (FN):** когда модель предсказывает отрицательный класс, но на самом деле это положительный.  
  Например, когда спам-сообщение не было распознано и попало в папку с обычной почтой.

### F-бета мера

В случае, если нужно больше уделить внимание **Precision** или **Recall**, можно использовать F-бета меру такого вида:

![Formula F1](./img/ml-score/image4.png)

- При **β < 1** большее внимание уделяется **Precision**.
- При **β > 1** большее внимание уделяется **Recall**.

---

# Метод Accuracy

**Accuracy** считает общее количество правильных ответов среди всей выборки, что плохо при неравномерных данных.  
Например, если в нашем датасете всего 600 наблюдений, из которых 550 — положительные, а 50 — отрицательные, и наша модель верно определила 530 положительных и всего 5 отрицательных, то общая Accuracy равна:
(530 + 5) / 600 = 0.8917


Это означает, что точность модели составляет **89.17%**.  
Полагаясь на это значение, можно подумать, что для любой выборки (независимо от ее класса) модель сделает правильный прогноз в 89.17% случаев. Однако это неверно, так как для класса **Negative** модель работает очень плохо.

---

# Метод Precision

**Precision** представляет собой отношение числа семплов, верно классифицированных как **Positive**, к общему числу выборок с меткой **Positive** (распознанных правильно и неправильно).  
Precision измеряет точность модели при определении класса **Positive**.

**Precision** фактически показывает процент доверия, что наблюдение относится к классу **Positive**. То есть сколько позитивных наблюдений действительно были определены как позитивные.  
Другими словами, Precision помогает оценить, оправдывается ли подавляющее большинство рекомендаций модели.

### Проблемы Precision:
- НЕ показывает охват позитивных наблюдений.  
  Например, Precision может выдать высокий результат для модели, которая называет положительными только заведомо очевидные наблюдения (например, "очевидно, что Илон Маск выплатит кредит за тостер").

---

# Метод Recall

**Recall** рассчитывается как отношение числа Positive выборок, корректно классифицированных как **Positive**, к общему количеству Positive семплов.  
Recall измеряет способность модели обнаруживать выборки, относящиеся к классу **Positive**. Чем выше Recall, тем больше Positive семплов было найдено.

**Recall** заботится только о том, как классифицируются Positive выборки. Эта метрика не зависит от того, как предсказываются Negative семплы, в отличие от Precision.  
Когда модель корректно классифицирует все Positive выборки, Recall будет **100%**, даже если все представители класса Negative были ошибочно определены как Positive.

### Пример:

На следующем изображении представлены 4 разных случая (от A до D), и все они имеют одинаковый Recall, равный **0.667**.  
Представленные примеры отличаются только тем, как классифицируются Negative семплы. Например, в случае **A** все Negative выборки корректно определены, а в случае **D** — наоборот.  
Независимо от того, как модель предсказывает класс Negative, Recall касается только семплов, относящихся к Positive.

![Recall Examples](./img/ml-score/image5.jpg)

---

# Precision или Recall?

Если мы хотим понять точность предсказания **Positive** и учесть в этой оценке те **Negative** наблюдения, которые модель пометила как Positive – нам нужно использовать **Precision**.

Если нам важно только, насколько хорошо модель предсказывает **Positive** наблюдения, и нам совсем не важно, сколько было ошибок в **Negative** наблюдениях – мы используем **Recall**.

#### ХОРОШАЯ МОДЕЛЬ ДОЛЖНА ПОКАЗЫВАТЬ ХОРОШИЕ ЗНАЧЕНИЯ ПО ОБОИМ ЭТИМ ВЫБОРКАМ.

---

# Метод "Один против всех" (One-vs-All или OvA)

Поскольку задача состоит в мультиклассовой классификации, нам нужно адаптировать метрики **F1**, **Precision**, **Recall**. Для этого мы посчитаем эти метрики отдельно для каждого класса, а потом просто возьмем их среднее.  
Если бы какой-то из классов был важнее, мы могли бы использовать для него вес (например, вес частоты определенного класса в датасете).  
Собственно, в этом и заключается суть метода "Один против всех".

### Макро-усреднение (Macro-averaging)

**Среднее значение метрик для всех классов, без учета их веса.**  
Все классы имеют одинаковую важность.

![Macro-Averaging Formula](./img/ml-score/image6.png)

Где:
- **K** — количество классов.

### Взвешенное усреднение (Weighted-averaging)

**Среднее значение метрик для всех классов с учетом их частоты (или веса).**  
Чем больше класс, тем больше его вклад в итоговую метрику.

![Weighted-Averaging Formula](./img/ml-score/image7.png)

Где:
- **nᵢ** — количество примеров в классе *i*.
- **K** — количество классов.

--- 

# Как все это реализовать?

Для решения задачи мультиклассовой классификации с несбалансированными данными, где важно учитывать все ошибки, можно использовать:

- **scikit-learn** — для вычисления метрик.
- **numpy** — для работы с массивами данных.

## Установка библиотек

```bash
pip install scikit-learn numpy
```

## Пример кода

```python
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix

# Для примера создадим фейковые данные
# Пусть у нас есть 1000 изображений (вектора признаков)
# 3 класса: 'A', 'B', 'C'

# Генерация случайных данных (1000 образцов, 10 признаков)
np.random.seed(42)
X = np.random.rand(1000, 10)

# Классы: A, B, C (с некоторым несбалансированным распределением)
y = np.array(['A']*700 + ['B']*200 + ['C']*100)

# Разделим данные на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Преобразуем метки классов в числа
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Обучим классификатор (например, случайный лес)
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train_encoded)

# Предсказания модели
y_pred = clf.predict(X_test)

# 1. Вычисление Precision, Recall и F1 с макро-усреднением:
precision_macro = precision_score(y_test_encoded, y_pred, average='macro')
recall_macro = recall_score(y_test_encoded, y_pred, average='macro')
f1_macro = f1_score(y_test_encoded, y_pred, average='macro')

print(f"Макро-усредненный Precision: {precision_macro:.4f}")
print(f"Макро-усредненный Recall: {recall_macro:.4f}")
print(f"Макро-усредненный F1: {f1_macro:.4f}")

# 2. Метод "Один против одного" (One-vs-One):
# Для этого мы используем 'average=None', чтобы вычислить метрики для каждой пары классов
precision_ovo = precision_score(y_test_encoded, y_pred, average=None)
recall_ovo = recall_score(y_test_encoded, y_pred, average=None)
f1_ovo = f1_score(y_test_encoded, y_pred, average=None)

print("\nМетрики Один против одного (One-vs-One) для каждого класса:")
print("Precision для каждого класса:", precision_ovo)
print("Recall для каждого класса:", recall_ovo)
print("F1 для каждого класса:", f1_ovo)

# 3. Дополнительно, можем вывести полный отчет по классификации (для всех классов)
print("\nПолный отчет по классификации:")
print(classification_report(y_test_encoded, y_pred, target_names=label_encoder.classes_))

# 4. Составим матрицу ошибок
conf_matrix = confusion_matrix(y_test_encoded, y_pred)
print("\nМатрица ошибок (Confusion Matrix):")
print(conf_matrix)
```
---

# Итог

В рамках работы над нашим проектом, исходя из того, что мы не сами собираем датасет и не можем гарантировать его сбалансированность, использование **F1-оценки** является самым оптимальным вариантом.  
В ином случае, мы не сможем оценить качество модели в полной мере.  

Поскольку у нас много классов, мы воспользуемся методом **"Один против всех"** для получения общей оценки модели.  
Кроме того, мы будем использовать библиотеку **scikit-learn** для применения готовых реализаций этих методов.

---

# Источники по метрикам

1. [Метрики Accuracy, Precision и Recall](https://pythonru.com/baza-znanij/metriki-accuracy-precision-i-recall)
2. [Habr: Метрики классификации](https://habr.com/ru/articles/661119/)
3. [F1-Score Glossary](https://www.ultralytics.com/ru/glossary/f1-score)
4. [Видеообзор на YouTube](https://www.youtube.com/watch?v=-jplpYLrcdM)
5. [ChatGPT](https://chatgpt.com)

