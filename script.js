// Управление группами данных
document.addEventListener('DOMContentLoaded', function() {
    const groupsContainer = document.getElementById('groups-container');
    const addGroupBtn = document.getElementById('add-group');
    const removeGroupBtn = document.getElementById('remove-group');
    
    if (addGroupBtn) {
        addGroupBtn.addEventListener('click', function() {
            const groupCount = document.querySelectorAll('.group-input').length;
            const newGroup = document.createElement('div');
            newGroup.className = 'group-input';
            newGroup.innerHTML = `
                <label>Группа ${groupCount + 1}:</label>
                <textarea class="group-data" rows="3" placeholder="Пример: 12.5, 14.3, 15.2, 13.8"></textarea>
            `;
            groupsContainer.appendChild(newGroup);
            
            // Активируем кнопку удаления, если групп больше 2
            if (groupCount + 1 > 2) {
                removeGroupBtn.disabled = false;
            }
        });
    }
    
    if (removeGroupBtn) {
        removeGroupBtn.addEventListener('click', function() {
            const groups = document.querySelectorAll('.group-input');
            if (groups.length > 2) {
                groups[groups.length - 1].remove();
                
                // Деактивируем кнопку удаления, если осталось 2 группы
                if (groups.length - 1 === 2) {
                    removeGroupBtn.disabled = true;
                }
            }
        });
    }
    
    // Кнопка "Рассчитать"
    const calculateBtn = document.getElementById('calculate-btn');
    if (calculateBtn) {
        calculateBtn.addEventListener('click', calculateCriteria);
    }
    
    // Кнопка "Загрузить пример"
    const exampleBtn = document.getElementById('example-btn');
    if (exampleBtn) {
        exampleBtn.addEventListener('click', loadExample);
    }
    
    // Кнопка "Очистить"
    const clearBtn = document.getElementById('clear-btn');
    if (clearBtn) {
        clearBtn.addEventListener('click', clearForm);
    }
});

// Загрузка примера данных
function loadExample() {
    if (typeof exampleData !== 'undefined' && exampleData[criteriaName]) {
        const data = exampleData[criteriaName];
        
        if (criteriaName in ['student', 'fisher', 'mann-whitney', 'spearman', 'pearson', 'siegel-tukey']) {
            document.getElementById('data1').value = data.data1;
            document.getElementById('data2').value = data.data2;
        } else if (criteriaName in ['cochran', 'bartlett', 'anova', 'hortley']) {
            const groups = document.querySelectorAll('.group-data');
            data.groups.forEach((groupData, index) => {
                if (groups[index]) {
                    groups[index].value = groupData;
                }
            });
        }
    }
}

// Очистка формы
function clearForm() {
    // Очищаем все текстовые поля
    document.querySelectorAll('textarea').forEach(textarea => {
        textarea.value = '';
    });
    
    // Скрываем результаты
    document.getElementById('results-section').style.display = 'none';
}

// Расчет критерия
async function calculateCriteria() {
    const calculateBtn = document.getElementById('calculate-btn');
    const originalText = calculateBtn.innerHTML;
    
    try {
        // Показываем индикатор загрузки
        calculateBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Вычисление...';
        calculateBtn.disabled = true;
        
        // Собираем данные в зависимости от критерия
        let requestData = {};
        
        if (criteriaName in ['student', 'fisher', 'mann-whitney', 'spearman', 'pearson', 'siegel-tukey']) {
            const data1 = document.getElementById('data1').value.trim();
            const data2 = document.getElementById('data2').value.trim();
            
            if (!data1 || !data2) {
                throw new Error('Пожалуйста, заполните оба поля с данными');
            }
            
            requestData = { data1, data2 };
        } else if (criteriaName in ['cochran', 'bartlett', 'anova', 'hortley']) {
            const groupElements = document.querySelectorAll('.group-data');
            const groups = [];
            
            groupElements.forEach((element, index) => {
                const value = element.value.trim();
                if (value) {
                    groups.push(value);
                }
            });
            
            if (groups.length < 2) {
                throw new Error('Пожалуйста, введите данные как минимум для двух групп');
            }
            
            requestData = { groups };
        }
        
        // Отправляем запрос на сервер
        const response = await fetch(`/calculate/${criteriaName}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(requestData)
        });
        
        const result = await response.json();
        
        if (result.error) {
            throw new Error(result.error);
        }
        
        // Отображаем результаты
        displayResults(result);
        
    } catch (error) {
        alert('Ошибка: ' + error.message);
        console.error('Ошибка расчета:', error);
    } finally {
        // Восстанавливаем кнопку
        calculateBtn.innerHTML = originalText;
        calculateBtn.disabled = false;
    }
}

// Отображение результатов
function displayResults(result) {
    const resultsSection = document.getElementById('results-section');
    const resultsContent = document.getElementById('results-content');
    const plotContainer = document.getElementById('plot-container');
    
    // Форматируем результаты
    let html = '<div class="results-grid">';
    
    for (const [key, value] of Object.entries(result.result)) {
        if (key === 'выбросы по группам') {
            html += `<div class="result-item">
                <span class="result-key">${key}:</span>
                <div class="result-value">`;
            
            value.forEach(group => {
                html += `<div>Группа ${group.группа}: ${group.выбросы.join(', ') || 'нет'} (${group.количество})</div>`;
            });
            
            html += `</div></div>`;
        } else {
            html += `<div class="result-item">
                <span class="result-key">${key}:</span>
                <span class="result-value">${value}</span>
            </div>`;
        }
    }
    
    html += '</div>';
    resultsContent.innerHTML = html;
    
    // Отображаем график, если он есть
    if (result.plot_url) {
        plotContainer.innerHTML = `<img src="${result.plot_url}" alt="Визуализация данных">`;
    } else {
        plotContainer.innerHTML = '<p>График недоступен</p>';
    }
    
    // Показываем секцию с результатами
    resultsSection.style.display = 'block';
    
    // Прокручиваем к результатам
    resultsSection.scrollIntoView({ behavior: 'smooth' });
}