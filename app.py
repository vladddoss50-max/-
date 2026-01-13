from flask import Flask, render_template, request, jsonify
import math
import numpy as np
from scipy import stats
import json

app = Flask(__name__)

# Табличные значения для различных критериев
CRITICAL_VALUES = {
    'fisher': {
        0.05: [
            [161.4, 199.5, 215.7, 224.6, 230.2, 234.0, 236.8, 238.9, 240.5, 241.9],
            [18.51, 19.00, 19.16, 19.25, 19.30, 19.33, 19.35, 19.37, 19.38, 19.40],
            [10.13, 9.55, 9.28, 9.12, 9.01, 8.94, 8.89, 8.85, 8.81, 8.79],
            [7.71, 6.94, 6.59, 6.39, 6.26, 6.16, 6.09, 6.04, 6.00, 5.96],
            [6.61, 5.79, 5.41, 5.19, 5.05, 4.95, 4.88, 4.82, 4.77, 4.74],
            [5.99, 5.14, 4.76, 4.53, 4.39, 4.28, 4.21, 4.15, 4.10, 4.06],
            [5.59, 4.74, 4.35, 4.12, 3.97, 3.87, 3.79, 3.73, 3.68, 3.64],
            [5.32, 4.46, 4.07, 3.84, 3.69, 3.58, 3.50, 3.44, 3.39, 3.35],
            [5.12, 4.26, 3.86, 3.63, 3.48, 3.37, 3.29, 3.23, 3.18, 3.14],
            [4.96, 4.10, 3.71, 3.48, 3.33, 3.22, 3.14, 3.07, 3.02, 2.98]
        ]
    },
    'student': {
        0.05: [12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228,
               2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086,
               2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042]
    },
    'cochran': {
        0.05: [
            [0.9985, 0.9750, 0.9392, 0.9057, 0.8772, 0.8534, 0.8332, 0.8159, 0.8010, 0.7880],
            [0.9669, 0.8709, 0.7977, 0.7457, 0.7071, 0.6771, 0.6530, 0.6333, 0.6167, 0.6025],
            [0.9065, 0.7679, 0.6841, 0.6287, 0.5892, 0.5598, 0.5365, 0.5175, 0.5017, 0.4884]
        ]
    },
    'bartlett': {
        0.05: [3.841, 5.991, 7.815, 9.488, 11.070, 12.592, 14.067, 15.507, 16.919, 18.307]
    }
}


def parse_data(data_str):
    """Парсинг данных из строки"""
    if not data_str:
        return []

    try:
        # Пробуем разные разделители
        for separator in [',', ';', ' ', '\n', '\t']:
            if separator in data_str:
                return [float(x.strip()) for x in data_str.split(separator) if x.strip()]

        # Если разделитель не найден, пробуем разбить по строкам
        return [float(x.strip()) for x in data_str.split('\n') if x.strip()]
    except ValueError:
        raise ValueError(
            "Ошибка в формате данных. Используйте числа, разделенные запятыми, пробелами или переносами строк")


def get_fisher_critical_value(df1, df2, alpha=0.05):
    """Получение критического значения критерия Фишера"""
    try:
        if df1 <= 10 and df2 <= 10:
            return CRITICAL_VALUES['fisher'][alpha][df2 - 1][df1 - 1]
        else:
            # Используем приближение для больших степеней свободы
            return stats.f.ppf(1 - alpha, df1, df2)
    except:
        return stats.f.ppf(1 - alpha, df1, df2)


def get_student_critical_value(df, alpha=0.05):
    """Получение критического значения критерия Стьюдента"""
    try:
        if df <= 30:
            return CRITICAL_VALUES['student'][alpha][df - 1]
        else:
            return stats.t.ppf(1 - alpha / 2, df)
    except:
        return stats.t.ppf(1 - alpha / 2, df)


def get_cochran_critical_value(k, n, alpha=0.05):
    """Получение критического значения критерия Кочрена"""
    try:
        if k <= 3 and n <= 10:
            return CRITICAL_VALUES['cochran'][alpha][k - 1][n - 1]
        else:
            # Приближенное значение
            return 1 / (1 + (k - 1) * stats.f.ppf(1 - alpha, n - 1, (n - 1) * (k - 1)))
    except:
        return stats.f.ppf(1 - alpha, n - 1, (n - 1) * (k - 1))


@app.route('/')
def index():
    """Главная страница с формой ввода"""
    return render_template('index.html')


@app.route('/solve', methods=['POST'])
def solve_problem():
    """Обработка решения задачи"""
    try:
        data = request.get_json()
        problem_type = data.get('problem_type', 'smirnov')

        result = {
            'type': problem_type,
            'problem': '',
            'solution': '',
            'details': [],
            'calculated_value': None,
            'critical_value': None,
            'hypothesis': ''
        }

        if problem_type == 'smirnov':
            # Критерий Смирнова
            data_str = data.get('data', '')
            x = parse_data(data_str)

            if len(x) < 3:
                return jsonify({'error': 'Объём выборки должен быть не менее 3'})

            n = len(x)
            x_sorted = sorted(x)
            mean_x = np.mean(x)

            # Дисперсия
            variance = np.var(x, ddof=1)
            std_dev = np.sqrt(variance) if variance > 0 else 0

            # Критерий Смирнова
            if std_dev > 0:
                u = abs(x_sorted[-1] - mean_x) / std_dev
            else:
                u = 0

            # Критические значения
            critical_values_smirnov = {
                5: 2.75, 6: 2.82, 7: 2.87, 8: 2.92, 9: 2.96, 10: 2.99,
                11: 3.03, 12: 3.06, 13: 3.09, 14: 3.12, 15: 3.14,
                16: 3.17, 17: 3.19, 18: 3.21, 19: 3.23, 20: 3.25,
                25: 3.31, 30: 3.35, 35: 3.39, 40: 3.42, 45: 3.45,
                50: 3.48, 100: 3.60, 200: 3.72
            }

            critical_value = critical_values_smirnov.get(n)
            if critical_value is None:
                # Интерполяция
                keys = sorted(critical_values_smirnov.keys())
                if n < keys[0]:
                    critical_value = critical_values_smirnov[keys[0]]
                elif n > keys[-1]:
                    critical_value = critical_values_smirnov[keys[-1]]
                else:
                    for i in range(len(keys) - 1):
                        if keys[i] <= n <= keys[i + 1]:
                            n1, cv1 = keys[i], critical_values_smirnov[keys[i]]
                            n2, cv2 = keys[i + 1], critical_values_smirnov[keys[i + 1]]
                            critical_value = cv1 + (cv2 - cv1) * (n - n1) / (n2 - n1)
                            break

            hypothesis = "Гипотеза о нормальности распределения ОТВЕРГАЕТСЯ (есть выбросы)" if u > critical_value else "Гипотеза о нормальности распределения ПРИНИМАЕТСЯ (выбросов нет)"

            result['problem'] = f'Критерий Смирнова (n={n})'
            result['solution'] = f'Эмпирическое значение: u = {u:.4f}'
            result['details'] = [
                f'Объём выборки: n = {n}',
                f'Среднее: {mean_x:.4f}',
                f'Дисперсия: {variance:.4f}',
                f'Стандартное отклонение: {std_dev:.4f}',
                f'Критическое значение (α=0.05): {critical_value:.4f}',
                hypothesis
            ]
            result['calculated_value'] = u
            result['critical_value'] = critical_value
            result['hypothesis'] = hypothesis

        elif problem_type == 'fisher':
            # Критерий Фишера (F-тест)
            data1_str = data.get('data1', '')
            data2_str = data.get('data2', '')

            x1 = parse_data(data1_str)
            x2 = parse_data(data2_str)

            if len(x1) < 2 or len(x2) < 2:
                return jsonify({'error': 'Каждая выборка должна содержать не менее 2 элементов'})

            # Вычисляем дисперсии
            var1 = np.var(x1, ddof=1)
            var2 = np.var(x2, ddof=1)

            # F-статистика (большая дисперсия / меньшая)
            if var1 >= var2:
                f_value = var1 / var2 if var2 != 0 else float('inf')
                df1, df2 = len(x1) - 1, len(x2) - 1
            else:
                f_value = var2 / var1 if var1 != 0 else float('inf')
                df1, df2 = len(x2) - 1, len(x1) - 1

            # Критическое значение
            critical_value = get_fisher_critical_value(df1, df2)

            hypothesis = "Гипотеза о равенстве дисперсий ОТВЕРГАЕТСЯ" if f_value > critical_value else "Гипотеза о равенстве дисперсий ПРИНИМАЕТСЯ"

            result['problem'] = 'F-тест (критерий Фишера)'
            result['solution'] = f'F-статистика: F = {f_value:.4f}'
            result['details'] = [
                f'Выборка 1: n₁ = {len(x1)}, дисперсия = {var1:.4f}',
                f'Выборка 2: n₂ = {len(x2)}, дисперсия = {var2:.4f}',
                f'Степени свободы: df₁ = {df1}, df₂ = {df2}',
                f'Критическое значение (α=0.05): {critical_value:.4f}',
                hypothesis
            ]
            result['calculated_value'] = f_value
            result['critical_value'] = critical_value
            result['hypothesis'] = hypothesis

        elif problem_type == 'student':
            # Критерий Стьюдента (t-тест)
            data1_str = data.get('data1', '')
            data2_str = data.get('data2', '')
            test_type = data.get('test_type', 'independent')

            x1 = parse_data(data1_str)
            x2 = parse_data(data2_str)

            if test_type == 'independent':
                # Двухвыборочный t-тест для независимых выборок
                if len(x1) < 2 or len(x2) < 2:
                    return jsonify({'error': 'Каждая выборка должна содержать не менее 2 элементов'})

                mean1, mean2 = np.mean(x1), np.mean(x2)
                var1, var2 = np.var(x1, ddof=1), np.var(x2, ddof=1)
                n1, n2 = len(x1), len(x2)

                # Объединенная дисперсия
                pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
                std_error = np.sqrt(pooled_var * (1 / n1 + 1 / n2))

                if std_error != 0:
                    t_value = abs(mean1 - mean2) / std_error
                else:
                    t_value = float('inf')

                df = n1 + n2 - 2

                result['problem'] = 'Двухвыборочный t-тест Стьюдента (независимые выборки)'
                result['details'] = [
                    f'Выборка 1: n₁ = {n1}, среднее = {mean1:.4f}, дисперсия = {var1:.4f}',
                    f'Выборка 2: n₂ = {n2}, среднее = {mean2:.4f}, дисперсия = {var2:.4f}',
                    f'Объединенная дисперсия: {pooled_var:.4f}'
                ]
            else:
                # Парный t-тест
                if len(x1) != len(x2):
                    return jsonify({'error': 'Для парного t-теста выборки должны быть одинакового размера'})

                differences = [x1[i] - x2[i] for i in range(len(x1))]
                mean_diff = np.mean(differences)
                std_diff = np.std(differences, ddof=1)

                if std_diff != 0:
                    t_value = abs(mean_diff) / (std_diff / np.sqrt(len(differences)))
                else:
                    t_value = float('inf')

                df = len(differences) - 1

                result['problem'] = 'Парный t-тест Стьюдента'
                result['details'] = [
                    f'Размер пар: n = {len(x1)}',
                    f'Средняя разность: {mean_diff:.4f}',
                    f'Стандартное отклонение разностей: {std_diff:.4f}'
                ]

            critical_value = get_student_critical_value(df)
            hypothesis = "Гипотеза о равенстве средних ОТВЕРГАЕТСЯ" if t_value > critical_value else "Гипотеза о равенстве средних ПРИНИМАЕТСЯ"

            result['solution'] = f't-статистика: t = {t_value:.4f}'
            result['details'].extend([
                f'Степени свободы: df = {df}',
                f'Критическое значение (α=0.05): {critical_value:.4f}',
                hypothesis
            ])
            result['calculated_value'] = t_value
            result['critical_value'] = critical_value
            result['hypothesis'] = hypothesis

        elif problem_type == 'siegel_tukey':
            # Критерий Сиджела-Тьюки
            data1_str = data.get('data1', '')
            data2_str = data.get('data2', '')

            x1 = parse_data(data1_str)
            x2 = parse_data(data2_str)

            if len(x1) < 2 or len(x2) < 2:
                return jsonify({'error': 'Каждая выборка должна содержать не менее 2 элементов'})

            # Объединяем и сортируем
            combined = x1 + x2
            n1, n2 = len(x1), len(x2)
            n = n1 + n2

            # Присваиваем ранги по методу Сиджела-Тьюки
            sorted_indices = sorted(range(n), key=lambda i: combined[i])
            ranks = [0] * n

            # Первый и последний получают ранг 1, второй и предпоследний - 2, и т.д.
            for i in range(n):
                if i % 2 == 0:
                    ranks[sorted_indices[i]] = i + 1
                else:
                    ranks[sorted_indices[n - i - 1]] = i + 1

            # Сумма рангов для первой выборки
            w1 = sum(ranks[:n1])

            # Ожидаемое значение и дисперсия
            expected_w = n1 * (n + 1) / 2
            var_w = n1 * n2 * (n + 1) / 12

            if var_w > 0:
                z_value = (w1 - expected_w) / np.sqrt(var_w)
            else:
                z_value = 0

            # Критическое значение для Z
            critical_value = 1.96  # для α=0.05

            hypothesis = "Гипотеза о равенстве дисперсий ОТВЕРГАЕТСЯ" if abs(
                z_value) > critical_value else "Гипотеза о равенстве дисперсий ПРИНИМАЕТСЯ"

            result['problem'] = 'Критерий Сиджела-Тьюки'
            result['solution'] = f'Z-статистика: Z = {z_value:.4f}'
            result['details'] = [
                f'Выборка 1: n₁ = {n1}',
                f'Выборка 2: n₂ = {n2}',
                f'Сумма рангов для выборки 1: W₁ = {w1}',
                f'Ожидаемая сумма рангов: E(W) = {expected_w:.4f}',
                f'Дисперсия: Var(W) = {var_w:.4f}',
                f'Критическое значение (α=0.05): ±{critical_value}',
                hypothesis
            ]
            result['calculated_value'] = z_value
            result['critical_value'] = critical_value
            result['hypothesis'] = hypothesis

        elif problem_type == 'mann_whitney':
            # Критерий Манна-Уитни
            data1_str = data.get('data1', '')
            data2_str = data.get('data2', '')

            x1 = parse_data(data1_str)
            x2 = parse_data(data2_str)

            if len(x1) < 2 or len(x2) < 2:
                return jsonify({'error': 'Каждая выборка должна содержать не менее 2 элементов'})

            # Объединяем и присваиваем ранги
            combined = x1 + x2
            n1, n2 = len(x1), len(x2)

            # Присваивание рангов с учетом связей
            from scipy.stats import rankdata
            ranks = rankdata(combined)

            # Сумма рангов для каждой выборки
            r1 = sum(ranks[:n1])
            r2 = sum(ranks[n1:])

            # Статистика U
            u1 = r1 - n1 * (n1 + 1) / 2
            u2 = r2 - n2 * (n2 + 1) / 2
            u_stat = min(u1, u2)

            # Z-статистика
            n = n1 + n2
            mean_u = n1 * n2 / 2
            var_u = n1 * n2 * (n + 1) / 12

            if var_u > 0:
                z_value = (u_stat - mean_u) / np.sqrt(var_u)
            else:
                z_value = 0

            critical_value = 1.96  # для α=0.05

            hypothesis = "Гипотеза о равенстве распределений ОТВЕРГАЕТСЯ" if abs(
                z_value) > critical_value else "Гипотеза о равенстве распределений ПРИНИМАЕТСЯ"

            result['problem'] = 'Критерий Манна-Уитни'
            result['solution'] = f'U-статистика: U = {u_stat:.4f}, Z = {z_value:.4f}'
            result['details'] = [
                f'Выборка 1: n₁ = {n1}, сумма рангов R₁ = {r1}',
                f'Выборка 2: n₂ = {n2}, сумма рангов R₂ = {r2}',
                f'U₁ = {u1:.4f}, U₂ = {u2:.4f}',
                f'Ожидаемое значение U: {mean_u:.4f}',
                f'Дисперсия: {var_u:.4f}',
                f'Критическое значение (α=0.05): ±{critical_value}',
                hypothesis
            ]
            result['calculated_value'] = u_stat
            result['z_value'] = z_value
            result['critical_value'] = critical_value
            result['hypothesis'] = hypothesis

        elif problem_type == 'spearman':
            # Критерий Спирмена
            x_str = data.get('x_data', '')
            y_str = data.get('y_data', '')

            x = parse_data(x_str)
            y = parse_data(y_str)

            if len(x) != len(y):
                return jsonify({'error': 'Ряды X и Y должны быть одинаковой длины'})

            if len(x) < 3:
                return jsonify({'error': 'Для корреляционного анализа нужно не менее 3 пар наблюдений'})

            n = len(x)

            # Вычисление рангов
            from scipy.stats import rankdata
            rx = rankdata(x)
            ry = rankdata(y)

            # Разности рангов
            d = [rx[i] - ry[i] for i in range(n)]

            # Коэффициент ранговой корреляции Спирмена
            sum_d2 = sum(di ** 2 for di in d)

            if n > 1:
                rho = 1 - (6 * sum_d2) / (n * (n ** 2 - 1))
            else:
                rho = 0

            # t-статистика для проверки значимости
            if abs(rho) < 1:
                t_value = rho * np.sqrt((n - 2) / (1 - rho ** 2))
            else:
                t_value = float('inf')

            df = n - 2
            critical_value = get_student_critical_value(df)

            hypothesis = "Корреляция СТАТИСТИЧЕСКИ ЗНАЧИМА" if abs(
                t_value) > critical_value else "Корреляция НЕ ЗНАЧИМА"

            result['problem'] = 'Ранговая корреляция Спирмена'
            result['solution'] = f'Коэффициент Спирмена: ρ = {rho:.4f}'
            result['details'] = [
                f'Количество пар: n = {n}',
                f'Сумма квадратов разностей рангов: Σd² = {sum_d2}',
                f't-статистика: t = {t_value:.4f}',
                f'Степени свободы: df = {df}',
                f'Критическое значение (α=0.05): ±{critical_value:.4f}',
                hypothesis,
                f'Сила связи: {"сильная" if abs(rho) > 0.7 else "средняя" if abs(rho) > 0.3 else "слабая"}'
            ]
            result['calculated_value'] = rho
            result['t_value'] = t_value
            result['critical_value'] = critical_value
            result['hypothesis'] = hypothesis

        elif problem_type == 'hortley':
            # Критерий Хортли (для выбросов)
            data_str = data.get('data', '')
            x = parse_data(data_str)

            if len(x) < 4:
                return jsonify({'error': 'Объём выборки должен быть не менее 4'})

            n = len(x)
            x_sorted = sorted(x)

            # Квартили
            q1_index = (n + 1) * 0.25
            q3_index = (n + 1) * 0.75

            q1 = np.percentile(x, 25)
            q3 = np.percentile(x, 75)
            iqr = q3 - q1

            # Границы для выбросов
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            # Поиск выбросов
            outliers = [val for val in x if val < lower_bound or val > upper_bound]

            result['problem'] = 'Критерий Хортли (выявление выбросов)'
            result['solution'] = f'Найдено выбросов: {len(outliers)}'
            result['details'] = [
                f'Объём выборки: n = {n}',
                f'Q₁ (25-й перцентиль): {q1:.4f}',
                f'Q₃ (75-й перцентиль): {q3:.4f}',
                f'Межквартильный размах: IQR = {iqr:.4f}',
                f'Нижняя граница: Q₁ - 1.5×IQR = {lower_bound:.4f}',
                f'Верхняя граница: Q₃ + 1.5×IQR = {upper_bound:.4f}',
                f'Выбросы: {outliers}' if outliers else 'Выбросов не обнаружено'
            ]
            result['outliers'] = outliers
            result['q1'] = q1
            result['q3'] = q3
            result['iqr'] = iqr

        elif problem_type == 'cochran':
            # Критерий Кочрена
            k = int(data.get('k', 2))  # количество групп
            n = int(data.get('n', 10))  # количество наблюдений в группе

            if k < 2:
                return jsonify({'error': 'Количество групп должно быть не менее 2'})
            if n < 2:
                return jsonify({'error': 'Количество наблюдений должно быть не менее 2'})

            # Генерация примерных данных или использование введенных
            variances = []
            for i in range(k):
                data_str = data.get(f'data{i + 1}', '')
                if data_str:
                    group_data = parse_data(data_str)
                    if len(group_data) < 2:
                        return jsonify({'error': f'Группа {i + 1} должна содержать не менее 2 элементов'})
                    var = np.var(group_data, ddof=1)
                else:
                    # Генерируем случайные данные
                    var = np.random.uniform(1, 10)
                variances.append(var)

            max_var = max(variances)
            sum_vars = sum(variances)

            # Статистика Кочрена
            g_value = max_var / sum_vars if sum_vars != 0 else 0

            # Критическое значение
            critical_value = get_cochran_critical_value(k, n)

            hypothesis = "Гипотеза об однородности дисперсий ОТВЕРГАЕТСЯ" if g_value > critical_value else "Гипотеза об однородности дисперсий ПРИНИМАЕТСЯ"

            result['problem'] = 'Критерий Кочрена'
            result['solution'] = f'Статистика Кочрена: G = {g_value:.4f}'
            result['details'] = [
                f'Количество групп: k = {k}',
                f'Объём выборок: n = {n}',
                f'Дисперсии групп: {[f"{v:.4f}" for v in variances]}',
                f'Максимальная дисперсия: {max_var:.4f}',
                f'Сумма дисперсий: {sum_vars:.4f}',
                f'Критическое значение (α=0.05): {critical_value:.4f}',
                hypothesis
            ]
            result['calculated_value'] = g_value
            result['critical_value'] = critical_value
            result['hypothesis'] = hypothesis

        elif problem_type == 'bartlett':
            # Критерий Бартлета
            k = int(data.get('k', 3))

            # Сбор данных для групп
            groups = []
            for i in range(k):
                data_str = data.get(f'data{i + 1}', '')
                if data_str:
                    group_data = parse_data(data_str)
                    if len(group_data) < 2:
                        return jsonify({'error': f'Группа {i + 1} должна содержать не менее 2 элементов'})
                    groups.append(group_data)
                else:
                    # Генерируем случайные данные
                    groups.append(np.random.normal(0, 1, 10).tolist())

            if k < 2:
                return jsonify({'error': 'Количество групп должно быть не менее 2'})

            # Вычисление статистики Бартлета
            n_values = [len(g) for g in groups]
            variances = [np.var(g, ddof=1) for g in groups]

            N = sum(n_values)
            k = len(groups)

            # Объединенная дисперсия
            pooled_var = sum([(n - 1) * var for n, var in zip(n_values, variances)]) / (N - k)

            # Статистика Бартлета
            chi2_numerator = (N - k) * np.log(pooled_var) - sum(
                [(n - 1) * np.log(var) for n, var in zip(n_values, variances)])
            chi2_denominator = 1 + (1 / (3 * (k - 1))) * (sum([1 / (n - 1) for n in n_values]) - 1 / (N - k))

            chi2_value = chi2_numerator / chi2_denominator if chi2_denominator != 0 else 0

            # Критическое значение
            df = k - 1
            try:
                if df <= 10:
                    critical_value = CRITICAL_VALUES['bartlett'][0.05][df - 1]
                else:
                    critical_value = stats.chi2.ppf(0.95, df)
            except:
                critical_value = stats.chi2.ppf(0.95, df)

            hypothesis = "Гипотеза об однородности дисперсий ОТВЕРГАЕТСЯ" if chi2_value > critical_value else "Гипотеза об однородности дисперсий ПРИНИМАЕТСЯ"

            result['problem'] = 'Критерий Бартлета'
            result['solution'] = f'χ²-статистика: χ² = {chi2_value:.4f}'
            result['details'] = [
                f'Количество групп: k = {k}',
                f'Общие наблюдения: N = {N}',
                f'Объём выборок: {n_values}',
                f'Дисперсии групп: {[f"{v:.4f}" for v in variances]}',
                f'Объединенная дисперсия: {pooled_var:.4f}',
                f'Степени свободы: df = {df}',
                f'Критическое значение (α=0.05): {critical_value:.4f}',
                hypothesis
            ]
            result['calculated_value'] = chi2_value
            result['critical_value'] = critical_value
            result['hypothesis'] = hypothesis

        elif problem_type == 'anova':
            # Однофакторный дисперсионный анализ
            k = int(data.get('k', 3))

            # Сбор данных для групп
            groups = []
            for i in range(k):
                data_str = data.get(f'data{i + 1}', '')
                if data_str:
                    group_data = parse_data(data_str)
                    if len(group_data) < 2:
                        return jsonify({'error': f'Группа {i + 1} должна содержать не менее 2 элементов'})
                    groups.append(group_data)
                else:
                    # Генерируем случайные данные
                    groups.append(np.random.normal(0, 1, 10).tolist())

            if k < 2:
                return jsonify({'error': 'Количество групп должно быть не менее 2'})

            # Вычисления для ANOVA
            n_values = [len(g) for g in groups]
            group_means = [np.mean(g) for g in groups]
            group_vars = [np.var(g, ddof=1) for g in groups]

            N = sum(n_values)
            overall_mean = sum([sum(g) for g in groups]) / N

            # Сумма квадратов
            ss_between = sum([n * (mean - overall_mean) ** 2 for n, mean in zip(n_values, group_means)])
            ss_within = sum([(n - 1) * var for n, var in zip(n_values, group_vars)])
            ss_total = ss_between + ss_within

            # Средние квадраты
            df_between = k - 1
            df_within = N - k
            df_total = N - 1

            ms_between = ss_between / df_between if df_between > 0 else 0
            ms_within = ss_within / df_within if df_within > 0 else 0

            # F-статистика
            f_value = ms_between / ms_within if ms_within > 0 else float('inf')

            # Критическое значение
            critical_value = get_fisher_critical_value(df_between, df_within)

            hypothesis = "Есть статистически значимые различия между группами" if f_value > critical_value else "Нет статистически значимых различий между группами"

            result['problem'] = 'Однофакторный дисперсионный анализ (ANOVA)'
            result['solution'] = f'F-статистика: F = {f_value:.4f}'
            result['details'] = [
                f'Количество групп: k = {k}',
                f'Общие наблюдения: N = {N}',
                f'Объём выборок: {n_values}',
                f'Средние по группам: {[f"{m:.4f}" for m in group_means]}',
                f'Общее среднее: {overall_mean:.4f}',
                f'SSмежду = {ss_between:.4f}, SSвнутри = {ss_within:.4f}, SSобщ = {ss_total:.4f}',
                f'MSмежду = {ms_between:.4f}, MSвнутри = {ms_within:.4f}',
                f'Степени свободы: df₁ = {df_between}, df₂ = {df_within}',
                f'Критическое значение (α=0.05): {critical_value:.4f}',
                hypothesis
            ]
            result['calculated_value'] = f_value
            result['critical_value'] = critical_value
            result['hypothesis'] = hypothesis

        elif problem_type == 'pearson':
            # Корреляция Пирсона
            x_str = data.get('x_data', '')
            y_str = data.get('y_data', '')

            x = parse_data(x_str)
            y = parse_data(y_str)

            if len(x) != len(y):
                return jsonify({'error': 'Ряды X и Y должны быть одинаковой длины'})

            if len(x) < 3:
                return jsonify({'error': 'Для корреляционного анализа нужно не менее 3 пар наблюдений'})

            n = len(x)

            # Коэффициент корреляции Пирсона
            mean_x, mean_y = np.mean(x), np.mean(y)
            std_x, std_y = np.std(x, ddof=1), np.std(y, ddof=1)

            if std_x > 0 and std_y > 0:
                covariance = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n)) / (n - 1)
                r_value = covariance / (std_x * std_y)
            else:
                r_value = 0

            # t-статистика для проверки значимости
            if abs(r_value) < 1:
                t_value = r_value * np.sqrt((n - 2) / (1 - r_value ** 2))
            else:
                t_value = float('inf')

            df = n - 2
            critical_value = get_student_critical_value(df)

            hypothesis = "Корреляция СТАТИСТИЧЕСКИ ЗНАЧИМА" if abs(
                t_value) > critical_value else "Корреляция НЕ ЗНАЧИМА"

            # Интерпретация силы связи
            abs_r = abs(r_value)
            if abs_r > 0.7:
                strength = "сильная"
            elif abs_r > 0.3:
                strength = "средняя"
            else:
                strength = "слабая"

            result['problem'] = 'Корреляция Пирсона'
            result['solution'] = f'Коэффициент корреляции Пирсона: r = {r_value:.4f}'
            result['details'] = [
                f'Количество пар: n = {n}',
                f'Среднее X: {mean_x:.4f}, Среднее Y: {mean_y:.4f}',
                f'Стандартное отклонение X: {std_x:.4f}',
                f'Стандартное отклонение Y: {std_y:.4f}',
                f'Ковариация: {covariance:.4f}' if 'covariance' in locals() else '',
                f't-статистика: t = {t_value:.4f}',
                f'Степени свободы: df = {df}',
                f'Критическое значение (α=0.05): ±{critical_value:.4f}',
                hypothesis,
                f'Сила связи: {strength}',
                f'Направление: {"положительная" if r_value > 0 else "отрицательная"}'
            ]
            result['calculated_value'] = r_value
            result['t_value'] = t_value
            result['critical_value'] = critical_value
            result['hypothesis'] = hypothesis

        else:
            return jsonify({'error': 'Неизвестный тип критерия'})

        return jsonify(result)

    except ValueError as e:
        return jsonify({'error': f'Ошибка ввода: {str(e)}'})
    except Exception as e:
        return jsonify({'error': f'Произошла ошибка: {str(e)}'})


@app.route('/generate_data', methods=['POST'])
def generate_data():
    """Генерация случайных данных"""
    try:
        data = request.get_json()
        criterion = data.get('criterion', 'smirnov')
        n = int(data.get('n', 10))

        np.random.seed()

        if criterion in ['smirnov', 'hortley']:
            # Одна выборка
            generated_data = np.random.normal(0, 1, n)
        elif criterion in ['fisher', 'student', 'siegel_tukey', 'mann_whitney']:
            # Две выборки
            n1 = int(data.get('n1', n))
            n2 = int(data.get('n2', n))
            generated_data = {
                'data1': np.random.normal(0, 1, n1).tolist(),
                'data2': np.random.normal(0.5, 1.2, n2).tolist()
            }
        elif criterion in ['spearman', 'pearson']:
            # Два коррелированных ряда
            x = np.random.normal(0, 1, n)
            noise = np.random.normal(0, 0.3, n)
            y = 0.7 * x + noise
            generated_data = {
                'x_data': x.tolist(),
                'y_data': y.tolist()
            }
        elif criterion in ['cochran', 'bartlett', 'anova']:
            # Несколько групп
            k = int(data.get('k', 3))
            groups = {}
            for i in range(k):
                group_data = np.random.normal(i * 0.5, 1 + i * 0.2, n)
                groups[f'data{i + 1}'] = group_data.tolist()
            generated_data = groups
        else:
            generated_data = np.random.normal(0, 1, n).tolist()

        # Округление
        if isinstance(generated_data, dict):
            for key in generated_data:
                if isinstance(generated_data[key], list):
                    generated_data[key] = [round(float(x), 3) for x in generated_data[key]]
        elif isinstance(generated_data, np.ndarray):
            generated_data = [round(float(x), 3) for x in generated_data]
        elif isinstance(generated_data, list):
            generated_data = [round(float(x), 3) for x in generated_data]

        return jsonify({
            'data': generated_data,
            'message': f'Сгенерированы данные для критерия {criterion}'
        })

    except Exception as e:
        return jsonify({'error': f'Ошибка генерации: {str(e)}'})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)