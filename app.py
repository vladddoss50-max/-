import numpy as np
import pandas as pd
from scipy import stats
from flask import Flask, render_template, request, jsonify, session
import json
import io
import base64
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

app = Flask(__name__)
app.secret_key = 'fipsi_secret_key_2024'


class StatisticsCalculator:
    """Класс для выполнения статистических расчетов"""

    @staticmethod
    def student_test(data1, data2):
        """Критерий Стьюдента для независимых выборок"""
        t_stat, p_value = stats.ttest_ind(data1, data2, nan_policy='omit')
        return {
            't-статистика': round(t_stat, 4),
            'p-значение': round(p_value, 4),
            'интерпретация': 'Различия статистически значимы' if p_value < 0.05 else 'Различия не статистически значимы'
        }

    @staticmethod
    def fisher_test(data1, data2):
        """Критерий Фишера (F-тест) для сравнения дисперсий"""
        var1 = np.var(data1, ddof=1)
        var2 = np.var(data2, ddof=1)
        f_value = var1 / var2 if var1 > var2 else var2 / var1
        df1 = len(data1) - 1
        df2 = len(data2) - 1
        p_value = 2 * min(stats.f.cdf(f_value, df1, df2), 1 - stats.f.cdf(f_value, df1, df2))
        return {
            'F-статистика': round(f_value, 4),
            'p-значение': round(p_value, 4),
            'интерпретация': 'Дисперсии различаются значимо' if p_value < 0.05 else 'Дисперсии не различаются значимо'
        }

    @staticmethod
    def mann_whitney_test(data1, data2):
        """Критерий Манна-Уитни"""
        u_stat, p_value = stats.mannwhitneyu(data1, data2, alternative='two-sided')
        return {
            'U-статистика': round(u_stat, 4),
            'p-значение': round(p_value, 4),
            'интерпретация': 'Различия статистически значимы' if p_value < 0.05 else 'Различия не статистически значимы'
        }

    @staticmethod
    def spearman_test(data1, data2):
        """Критерий Спирмена"""
        corr, p_value = stats.spearmanr(data1, data2)
        return {
            'Коэффициент корреляции Спирмена': round(corr, 4),
            'p-значение': round(p_value, 4),
            'интерпретация': 'Корреляция статистически значима' if p_value < 0.05 else 'Корреляция не статистически значима'
        }

    @staticmethod
    def cochran_test(data_array):
        """Критерий Кочрена"""
        # Преобразуем данные в массив, где каждая строка - группа
        data = np.array(data_array)
        # Критерий Кочрена применяется к бинарным данным
        # Для демонстрации используем упрощенный подход
        c_stat, p_value = stats.chisquare([np.sum(row) for row in data])
        return {
            'Статистика критерия': round(c_stat, 4),
            'p-значение': round(p_value, 4),
            'интерпретация': 'Различия между группами статистически значимы' if p_value < 0.05 else 'Различия не статистически значимы'
        }

    @staticmethod
    def bartlett_test(data_array):
        """Критерий Бартлета на равенство дисперсий"""
        stat, p_value = stats.bartlett(*data_array)
        return {
            'Статистика критерия': round(stat, 4),
            'p-значение': round(p_value, 4),
            'интерпретация': 'Дисперсии различаются значимо' if p_value < 0.05 else 'Дисперсии не различаются значимо'
        }

    @staticmethod
    def anova_one_way(data_array):
        """Однофакторный дисперсионный анализ"""
        f_stat, p_value = stats.f_oneway(*data_array)
        return {
            'F-статистика': round(f_stat, 4),
            'p-значение': round(p_value, 4),
            'интерпретация': 'Есть статистически значимые различия между группами' if p_value < 0.05 else 'Нет статистически значимых различий между группами'
        }

    @staticmethod
    def pearson_correlation(data1, data2):
        """Корреляция Пирсона"""
        corr, p_value = stats.pearsonr(data1, data2)
        return {
            'Коэффициент корреляции Пирсона': round(corr, 4),
            'p-значение': round(p_value, 4),
            'интерпретация': 'Корреляция статистически значима' if p_value < 0.05 else 'Корреляция не статистически значима'
        }

    @staticmethod
    def siegel_tukey_test(data1, data2):
        """Критерий Сиджела-Тьюки"""
        # Объединяем выборки и ранжируем
        combined = np.concatenate([data1, data2])
        ranks = stats.rankdata(combined)

        # Присваиваем веса по методу Сиджела-Тьюки
        n = len(combined)
        weights = []
        for i in range(1, n + 1):
            if i % 2 == 0:
                weights.append(i)
            else:
                weights.append(n - i + 1)

        # Сортируем веса в порядке рангов
        sorted_weights = [weights[int(r - 1)] for r in ranks]

        # Разделяем веса обратно на две группы
        w1 = sorted_weights[:len(data1)]
        w2 = sorted_weights[len(data1):]

        # Применяем критерий Манна-Уитни к весам
        u_stat, p_value = stats.mannwhitneyu(w1, w2, alternative='two-sided')

        return {
            'U-статистика': round(u_stat, 4),
            'p-значение': round(p_value, 4),
            'интерпретация': 'Различия в дисперсиях статистически значимы' if p_value < 0.05 else 'Различия в дисперсиях не статистически значимы'
        }

    @staticmethod
    def hortley_test(data_array):
        """Критерий Хортлея (для выбросов)"""
        # Упрощенная версия критерия Хортлея
        all_data = np.concatenate(data_array)
        q1 = np.percentile(all_data, 25)
        q3 = np.percentile(all_data, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outliers = []
        for i, group in enumerate(data_array):
            group_outliers = [val for val in group if val < lower_bound or val > upper_bound]
            outliers.append({
                'группа': i + 1,
                'выбросы': group_outliers,
                'количество': len(group_outliers)
            })

        return {
            'границы выбросов': [round(lower_bound, 4), round(upper_bound, 4)],
            'выбросы по группам': outliers
        }


@app.route('/')
def index():
    """Главная страница"""
    return render_template('index.html')


@app.route('/criteria/<criteria_name>')
def criteria_page(criteria_name):
    """Страница с формой для ввода данных"""
    criteria_info = {
        'fisher': {'name': 'Критерий Фишера', 'description': 'Сравнение дисперсий двух выборок'},
        'student': {'name': 'Критерий Стьюдента', 'description': 'Сравнение средних двух выборок'},
        'siegel-tukey': {'name': 'Критерий Сиджела-Тьюки',
                         'description': 'Сравнение дисперсий двух выборок (непараметрический)'},
        'mann-whitney': {'name': 'Критерий Манна-Уитни',
                         'description': 'Сравнение двух независимых выборок (непараметрический)'},
        'spearman': {'name': 'Критерий Спирмена', 'description': 'Корреляционный анализ (непараметрический)'},
        'hortley': {'name': 'Критерий Хортлея', 'description': 'Обнаружение выбросов в данных'},
        'cochran': {'name': 'Критерий Кочрена', 'description': 'Сравнение нескольких долей'},
        'bartlett': {'name': 'Критерий Бартлета', 'description': 'Проверка равенства дисперсий нескольких выборок'},
        'anova': {'name': 'Однофакторный дисперсионный анализ', 'description': 'Сравнение средних нескольких выборок'},
        'pearson': {'name': 'Корреляция Пирсона', 'description': 'Оценка линейной связи между переменными'}
    }

    if criteria_name not in criteria_info:
        return render_template('index.html')

    return render_template('criteria.html',
                           criteria_name=criteria_name,
                           criteria=criteria_info[criteria_name])


@app.route('/calculate/<criteria_name>', methods=['POST'])
def calculate(criteria_name):
    """Обработка расчета статистического критерия"""
    try:
        data = request.get_json()
        calculator = StatisticsCalculator()

        if criteria_name == 'student':
            data1 = [float(x) for x in data['data1'].split(',')]
            data2 = [float(x) for x in data['data2'].split(',')]
            result = calculator.student_test(data1, data2)

        elif criteria_name == 'fisher':
            data1 = [float(x) for x in data['data1'].split(',')]
            data2 = [float(x) for x in data['data2'].split(',')]
            result = calculator.fisher_test(data1, data2)

        elif criteria_name == 'mann-whitney':
            data1 = [float(x) for x in data['data1'].split(',')]
            data2 = [float(x) for x in data['data2'].split(',')]
            result = calculator.mann_whitney_test(data1, data2)

        elif criteria_name == 'spearman':
            data1 = [float(x) for x in data['data1'].split(',')]
            data2 = [float(x) for x in data['data2'].split(',')]
            result = calculator.spearman_test(data1, data2)

        elif criteria_name == 'pearson':
            data1 = [float(x) for x in data['data1'].split(',')]
            data2 = [float(x) for x in data['data2'].split(',')]
            result = calculator.pearson_correlation(data1, data2)

        elif criteria_name == 'siegel-tukey':
            data1 = [float(x) for x in data['data1'].split(',')]
            data2 = [float(x) for x in data['data2'].split(',')]
            result = calculator.siegel_tukey_test(data1, data2)

        elif criteria_name == 'cochran':
            groups = []
            for group_data in data['groups']:
                groups.append([float(x) for x in group_data.split(',')])
            result = calculator.cochran_test(groups)

        elif criteria_name == 'bartlett':
            groups = []
            for group_data in data['groups']:
                groups.append([float(x) for x in group_data.split(',')])
            result = calculator.bartlett_test(groups)

        elif criteria_name == 'anova':
            groups = []
            for group_data in data['groups']:
                groups.append([float(x) for x in group_data.split(',')])
            result = calculator.anova_one_way(groups)

        elif criteria_name == 'hortley':
            groups = []
            for group_data in data['groups']:
                groups.append([float(x) for x in group_data.split(',')])
            result = calculator.hortley_test(groups)

        else:
            return jsonify({'error': 'Неизвестный критерий'})

        # Создание графика для визуализации данных
        plot_url = create_plot(criteria_name, data)

        return jsonify({
            'success': True,
            'result': result,
            'plot_url': plot_url
        })

    except Exception as e:
        return jsonify({'error': str(e)})


def create_plot(criteria_name, data):
    """Создание графика для визуализации данных"""
    plt.figure(figsize=(10, 6))

    try:
        if criteria_name in ['student', 'fisher', 'mann-whitney', 'spearman', 'pearson', 'siegel-tukey']:
            data1 = [float(x) for x in data['data1'].split(',')]
            data2 = [float(x) for x in data['data2'].split(',')]

            plt.subplot(1, 2, 1)
            plt.boxplot([data1, data2])
            plt.xticks([1, 2], ['Выборка 1', 'Выборка 2'])
            plt.title('Диаграмма размаха')
            plt.grid(True, alpha=0.3)

            plt.subplot(1, 2, 2)
            plt.hist(data1, alpha=0.5, label='Выборка 1', bins=10)
            plt.hist(data2, alpha=0.5, label='Выборка 2', bins=10)
            plt.title('Гистограмма распределений')
            plt.legend()
            plt.grid(True, alpha=0.3)

        elif criteria_name in ['cochran', 'bartlett', 'anova', 'hortley']:
            groups = []
            for group_data in data['groups']:
                groups.append([float(x) for x in group_data.split(',')])

            plt.boxplot(groups)
            plt.title('Диаграмма размаха по группам')
            plt.xlabel('Группы')
            plt.ylabel('Значения')
            plt.grid(True, alpha=0.3)

        plt.tight_layout()

        # Сохранение графика в base64
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close()

        return f"data:image/png;base64,{plot_url}"

    except Exception as e:
        plt.close()
        return None


if __name__ == '__main__':
    app.run(debug=True)