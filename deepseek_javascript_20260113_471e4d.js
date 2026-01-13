// Библиотека статистических функций на чистом JavaScript

class FIPSIStatistics {
    // Критерий Смирнова
    static smirnov(data) {
        const n = data.length;
        const sorted = [...data].sort((a, b) => a - b);
        const mean = sorted.reduce((a, b) => a + b) / n;
        const variance = sorted.reduce((sum, x) => sum + Math.pow(x - mean, 2), 0) / (n - 1);
        const std = Math.sqrt(variance);
        const u = Math.abs(sorted[sorted.length - 1] - mean) / std;
        
        // Критические значения
        const criticalValues = {
            5: 2.75, 6: 2.82, 7: 2.87, 8: 2.92, 9: 2.96, 10: 2.99,
            11: 3.03, 12: 3.06, 13: 3.09, 14: 3.12, 15: 3.14,
            16: 3.17, 17: 3.19, 18: 3.21, 19: 3.23, 20: 3.25,
            25: 3.31, 30: 3.35, 35: 3.39, 40: 3.42, 45: 3.45,
            50: 3.48, 100: 3.60, 200: 3.72
        };
        
        let critical = criticalValues[n] || 3.0;
        const hypothesis = u > critical ? 
            "Гипотеза о нормальности ОТВЕРГАЕТСЯ" : 
            "Гипотеза о нормальности ПРИНИМАЕТСЯ";
        
        return {
            u, critical, hypothesis,
            details: [
                `Объём выборки: n = ${n}`,
                `Среднее: ${mean.toFixed(4)}`,
                `Дисперсия: ${variance.toFixed(4)}`,
                `Стандартное отклонение: ${std.toFixed(4)}`,
                `Критическое значение (α=0.05): ${critical.toFixed(4)}`
            ]
        };
    }
    
    // Критерий Фишера
    static fisher(data1, data2) {
        const n1 = data1.length, n2 = data2.length;
        const mean1 = data1.reduce((a, b) => a + b) / n1;
        const mean2 = data2.reduce((a, b) => a + b) / n2;
        const var1 = data1.reduce((s, x) => s + Math.pow(x - mean1, 2), 0) / (n1 - 1);
        const var2 = data2.reduce((s, x) => s + Math.pow(x - mean2, 2), 0) / (n2 - 1);
        
        const F = var1 > var2 ? var1 / var2 : var2 / var1;
        
        // Табличные значения Фишера (α=0.05)
        const fTable = {
            1: {1: 161.4, 2: 18.51, 3: 10.13, 4: 7.71, 5: 6.61},
            2: {1: 199.5, 2: 19.00, 3: 9.55, 4: 6.94, 5: 5.79},
            // ... продолжение таблицы
        };
        
        const df1 = n1 - 1, df2 = n2 - 1;
        const critical = fTable[df1] && fTable[df1][df2] ? fTable[df1][df2] : 4.0;
        
        const hypothesis = F > critical ? 
            "Гипотеза о равенстве дисперсий ОТВЕРГАЕТСЯ" : 
            "Гипотеза о равенстве дисперсий ПРИНИМАЕТСЯ";
        
        return {
            F, critical, hypothesis,
            details: [
                `Выборка 1: n=${n1}, дисперсия=${var1.toFixed(4)}`,
                `Выборка 2: n=${n2}, дисперсия=${var2.toFixed(4)}`,
                `Степени свободы: df1=${df1}, df2=${df2}`,
                `Критическое значение: ${critical.toFixed(4)}`
            ]
        };
    }
    
    // Критерий Стьюдента
    static student(data1, data2, type = 'independent') {
        const n1 = data1.length, n2 = data2.length;
        const mean1 = data1.reduce((a, b) => a + b) / n1;
        const mean2 = data2.reduce((a, b) => a + b) / n2;
        
        let t, df;
        
        if (type === 'independent') {
            const var1 = data1.reduce((s, x) => s + Math.pow(x - mean1, 2), 0) / (n1 - 1);
            const var2 = data2.reduce((s, x) => s + Math.pow(x - mean2, 2), 0) / (n2 - 1);
            
            const pooledVar = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2);
            const se = Math.sqrt(pooledVar * (1/n1 + 1/n2));
            
            t = Math.abs(mean1 - mean2) / se;
            df = n1 + n2 - 2;
        } else {
            // Парный тест
            const diffs = data1.map((x, i) => x - data2[i]);
            const meanDiff = diffs.reduce((a, b) => a + b) / n1;
            const varDiff = diffs.reduce((s, d) => s + Math.pow(d - meanDiff, 2), 0) / (n1 - 1);
            const seDiff = Math.sqrt(varDiff / n1);
            
            t = Math.abs(meanDiff) / seDiff;
            df = n1 - 1;
        }
        
        // Таблица Стьюдента (α=0.05, двусторонний)
        const tTable = {
            1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
            6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
            15: 2.131, 20: 2.086, 30: 2.042, 60: 2.000, 120: 1.980
        };
        
        let critical = tTable[df];
        if (!critical) {
            // Аппроксимация для больших df
            critical = 1.96 + (2.375 / (df - 1.143));
        }
        
        const hypothesis = t > critical ? 
            "Гипотеза о равенстве средних ОТВЕРГАЕТСЯ" : 
            "Гипотеза о равенстве средних ПРИНИМАЕТСЯ";
        
        return {
            t, critical, hypothesis, df,
            details: [
                `Выборка 1: n=${n1}, среднее=${mean1.toFixed(4)}`,
                `Выборка 2: n=${n2}, среднее=${mean2.toFixed(4)}`,
                `Степени свободы: df=${df}`,
                `Критическое значение (α=0.05): ${critical.toFixed(4)}`
            ]
        };
    }
    
    // Корреляция Пирсона
    static pearsonCorrelation(x, y) {
        const n = x.length;
        const meanX = x.reduce((a, b) => a + b) / n;
        const meanY = y.reduce((a, b) => a + b) / n;
        
        let numerator = 0, denomX = 0, denomY = 0;
        
        for (let i = 0; i < n; i++) {
            const dx = x[i] - meanX;
            const dy = y[i] - meanY;
            numerator += dx * dy;
            denomX += dx * dx;
            denomY += dy * dy;
        }
        
        const r = numerator / Math.sqrt(denomX * denomY);
        const t = r * Math.sqrt((n - 2) / (1 - r * r));
        const df = n - 2;
        
        // Критическое значение Стьюдента
        const critical = this.getCriticalT(df);
        
        const hypothesis = Math.abs(t) > critical ? 
            "Корреляция СТАТИСТИЧЕСКИ ЗНАЧИМА" : 
            "Корреляция НЕ ЗНАЧИМА";
        
        let strength = "слабая";
        if (Math.abs(r) > 0.7) strength = "сильная";
        else if (Math.abs(r) > 0.3) strength = "средняя";
        
        return {
            r, t, critical, hypothesis, strength, df,
            details: [
                `Количество пар: n = ${n}`,
                `Коэффициент корреляции: r = ${r.toFixed(4)}`,
                `t-статистика: t = ${t.toFixed(4)}`,
                `Степени свободы: df = ${df}`,
                `Критическое значение: ${critical.toFixed(4)}`,
                `Сила связи: ${strength}`,
                `Направление: ${r > 0 ? "положительная" : "отрицательная"}`
            ]
        };
    }
    
    // Критерий Хортли (выбросы)
    static hortley(data) {
        const sorted = [...data].sort((a, b) => a - b);
        const n = sorted.length;
        
        // Квартили
        const q1 = this.percentile(sorted, 25);
        const q3 = this.percentile(sorted, 75);
        const iqr = q3 - q1;
        
        const lowerBound = q1 - 1.5 * iqr;
        const upperBound = q3 + 1.5 * iqr;
        
        const outliers = sorted.filter(x => x < lowerBound || x > upperBound);
        
        return {
            q1, q3, iqr, lowerBound, upperBound, outliers,
            details: [
                `Объём выборки: n = ${n}`,
                `Q₁ (25-й перцентиль): ${q1.toFixed(4)}`,
                `Q₃ (75-й перцентиль): ${q3.toFixed(4)}`,
                `Межквартильный размах: IQR = ${iqr.toFixed(4)}`,
                `Нижняя граница: ${lowerBound.toFixed(4)}`,
                `Верхняя граница: ${upperBound.toFixed(4)}`,
                outliers.length > 0 ? 
                    `Выбросы: ${outliers.map(o => o.toFixed(2)).join(', ')}` :
                    'Выбросов не обнаружено'
            ]
        };
    }
    
    // Вспомогательные методы
    static percentile(arr, p) {
        const index = (arr.length - 1) * p / 100;
        const lower = Math.floor(index);
        const upper = Math.ceil(index);
        
        if (lower === upper) return arr[lower];
        
        const weight = index - lower;
        return arr[lower] * (1 - weight) + arr[upper] * weight;
    }
    
    static getCriticalT(df) {
        const tTable = {
            1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
            6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
            15: 2.131, 20: 2.086, 30: 2.042, 60: 2.000, 120: 1.980
        };
        
        if (tTable[df]) return tTable[df];
        if (df > 120) return 1.96;
        
        // Интерполяция
        const keys = Object.keys(tTable).map(Number).sort((a, b) => a - b);
        for (let i = 0; i < keys.length - 1; i++) {
            if (df >= keys[i] && df <= keys[i + 1]) {
                const t1 = tTable[keys[i]];
                const t2 = tTable[keys[i + 1]];
                return t1 + (t2 - t1) * (df - keys[i]) / (keys[i + 1] - keys[i]);
            }
        }
        
        return 2.0;
    }
    
    // Генерация случайных данных
    static generateData(type, n = 10, params = {}) {
        const data = [];
        
        switch(type) {
            case 'normal':
                for (let i = 0; i < n; i++) {
                    data.push(this.normalRandom());
                }
                break;
            case 'uniform':
                for (let i = 0; i < n; i++) {
                    data.push(Math.random() * 10);
                }
                break;
            case 'correlated':
                const x = [];
                for (let i = 0; i < n; i++) {
                    x.push(Math.random() * 10);
                }
                return {
                    x: x,
                    y: x.map(val => val * 0.7 + Math.random() * 2)
                };
        }
        
        return data;
    }
    
    static normalRandom() {
        let u = 0, v = 0;
        while (u === 0) u = Math.random();
        while (v === 0) v = Math.random();
        return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
    }
}

// Экспорт для использования в браузере
window.FIPSIStatistics = FIPSIStatistics;