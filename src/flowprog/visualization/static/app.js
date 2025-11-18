// FlowProg Visualization Application

let cy; // Cytoscape instance
let graphData = null;
let stepsData = [];
let currentStep = -1;
let isPlaying = false;
let playInterval = null;
let savedPositions = null; // Save node positions after first layout
let selectedItem = null; // Track selected item: {type: 'process'|'flow', id: string, ...}
let globalEdgeMinMax = null; // Global min/max for edge width scaling

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    console.log('DOM Content Loaded');
    initializeCytoscape();
    loadStepsData();
});

// Initialize Cytoscape graph
function initializeCytoscape() {
    cy = cytoscape({
        container: document.getElementById('cy'),

        style: [
            // Process nodes
            {
                selector: 'node.process',
                style: {
                    'background-color': '#3498db',
                    'label': 'data(label)',
                    'color': '#fff',
                    'text-valign': 'center',
                    'text-halign': 'center',
                    'font-size': '12px',
                    'font-weight': 'bold',
                    'width': '80px',
                    'height': '60px',
                    'shape': 'rectangle',
                    'border-width': 2,
                    'border-color': '#2980b9',
                    'text-wrap': 'wrap',
                    'text-max-width': '75px'
                }
            },
            // Object nodes
            {
                selector: 'node.object',
                style: {
                    'background-color': '#95a5a6',
                    'label': 'data(label)',
                    'color': '#2c3e50',
                    'text-valign': 'center',
                    'text-halign': 'center',
                    'font-size': '11px',
                    'width': '60px',
                    'height': '60px',
                    'shape': 'ellipse',
                    'border-width': 2,
                    'border-color': '#7f8c8d',
                    'text-wrap': 'wrap',
                    'text-max-width': '55px'
                }
            },
            // Market objects (highlighted)
            {
                selector: 'node.market',
                style: {
                    'background-color': '#e74c3c',
                    'border-color': '#c0392b',
                    'color': '#fff'
                }
            },
            // Flow edges
            {
                selector: 'edge.flow',
                style: {
                    'width': 3,
                    'line-color': '#34495e',
                    'target-arrow-color': '#34495e',
                    'target-arrow-shape': 'triangle',
                    'curve-style': 'bezier',
                    'label': 'data(label)',
                    'font-size': '10px',
                    'text-rotation': 'autorotate',
                    'text-margin-y': -10,
                    'color': '#2c3e50',
                    'text-background-color': '#fff',
                    'text-background-opacity': 0.8,
                    'text-background-padding': '3px',
                    'text-wrap': 'wrap'
                }
            },
            // Selected elements
            {
                selector: ':selected',
                style: {
                    'border-width': 4,
                    'border-color': '#f39c12',
                    'line-color': '#f39c12',
                    'target-arrow-color': '#f39c12',
                    'z-index': 999
                }
            },
            // Hover effect
            {
                selector: 'node:active',
                style: {
                    'overlay-color': '#f39c12',
                    'overlay-padding': 8,
                    'overlay-opacity': 0.3
                }
            },
            {
                selector: 'edge:active',
                style: {
                    'overlay-color': '#f39c12',
                    'overlay-padding': 4,
                    'overlay-opacity': 0.3
                }
            }
        ],

        layout: {
            name: 'dagre',
            rankDir: 'LR',
            nodeSep: 50,
            rankSep: 100,
            padding: 30
        }
    });

    // Add click handlers
    cy.on('tap', 'node.process', function(evt) {
        const node = evt.target;
        console.log('Process node clicked:', node.data('id'));
        loadProcessDetails(node.data('id'));
    });

    cy.on('tap', 'edge.flow', function(evt) {
        const edge = evt.target;
        const source = edge.data('source');
        const target = edge.data('target');
        const material = edge.data('material');
        console.log('Flow edge clicked:', source, '->', target, '(', material, ')');
        loadFlowDetails(source, target, material);
    });

    // Click on background to clear selection
    cy.on('tap', function(evt) {
        if (evt.target === cy) {
            console.log('Background clicked, clearing selection');
            clearSelection();
        }
    });

    console.log('Cytoscape initialized successfully');
}

// Load graph data from API
async function loadGraphData() {
    try {
        console.log('Loading graph data from API...');
        const response = await fetch('/api/graph');
        graphData = await response.json();
        console.log('Graph data loaded:', graphData.nodes.length, 'nodes,', graphData.edges.length, 'edges');

        // Add elements to graph
        cy.add(graphData.nodes);
        cy.add(graphData.edges);
        console.log('Elements added to Cytoscape');

        // Run layout
        cy.layout({
            name: 'dagre',
            rankDir: 'LR',
            nodeSep: 50,
            rankSep: 100,
            padding: 30
        }).run();

        // Fit to screen
        cy.fit(null, 50);
        console.log('Graph layout complete');

    } catch (error) {
        console.error('Error loading graph data:', error);
        showError('Failed to load graph data');
    }
}

// Load process details
async function loadProcessDetails(processId) {
    try {
        console.log(`Fetching process details for: ${processId}`);
        const response = await fetch(`/api/process/${encodeURIComponent(processId)}`);
        const data = await response.json();
        console.log('Process details received:', data);

        if (data.error) {
            console.error('API returned error:', data.error);
            showError(data.error);
            return;
        }

        displayProcessDetails(data);

    } catch (error) {
        console.error('Error loading process details:', error);
        showError('Failed to load process details');
    }
}

// Load flow details
async function loadFlowDetails(source, target, material) {
    try {
        const response = await fetch(`/api/flow/${encodeURIComponent(source)}/${encodeURIComponent(target)}/${encodeURIComponent(material)}`);
        const data = await response.json();

        if (data.error) {
            showError(data.error);
            return;
        }

        displayFlowDetails(data);

    } catch (error) {
        console.error('Error loading flow details:', error);
        showError('Failed to load flow details');
    }
}

// Display process details in panel
function displayProcessDetails(data) {
    console.log('Displaying process details...');
    const panel = document.getElementById('details-panel');
    const title = document.getElementById('panel-title');
    const content = document.getElementById('panel-content');

    if (!panel || !title || !content) {
        console.error('Panel elements not found!', {panel, title, content});
        return;
    }

    title.textContent = `Process: ${data.process_id}`;

    let html = `
        <div class="section">
            <div class="section-title">Process Information</div>
            <div class="info-grid">
                <div class="info-label">ID:</div>
                <div class="info-value">${data.process_id}</div>
                <div class="info-label">Has Stock:</div>
                <div class="info-value">${data.has_stock ? 'Yes' : 'No'}</div>
                <div class="info-label">Consumes:</div>
                <div class="info-value">${data.consumes.join(', ')}</div>
                <div class="info-label">Produces:</div>
                <div class="info-value">${data.produces.join(', ')}</div>
            </div>
        </div>
    `;

    // Input flows
    if (data.inputs.length > 0) {
        html += `
            <div class="section">
                <div class="section-title">Input Flows</div>
        `;
        for (const input of data.inputs) {
            html += `
                <div class="flow-item">
                    <div class="flow-object">${input.object}</div>
                    <div class="flow-value">Value: ${input.value}</div>
                </div>
            `;
        }
        html += `</div>`;
    }

    // Output flows
    if (data.outputs.length > 0) {
        html += `
            <div class="section">
                <div class="section-title">Output Flows</div>
        `;
        for (const output of data.outputs) {
            html += `
                <div class="flow-item">
                    <div class="flow-object">${output.object}</div>
                    <div class="flow-value">Value: ${output.value}</div>
                </div>
            `;
        }
        html += `</div>`;
    }

    // X and Y expressions
    // If has_stock is false, X = Y, so show them once
    if (!data.has_stock && data.x_analysis.final_expression === data.y_analysis.final_expression) {
        html += `
            <div class="section">
                <div class="section-title">Process Activity (X = Y, no stock)</div>
                <p style="font-size: 0.85rem; color: #7f8c8d; margin-bottom: 0.75rem;">
                    This process has no stock, so input magnitude (X) equals output magnitude (Y).
                </p>
                ${renderExpression(data.x_analysis)}
            </div>
        `;
    } else {
        // Show X and Y separately
        html += `
            <div class="section">
                <div class="section-title">X (Process Input Magnitude)</div>
                ${renderExpression(data.x_analysis)}
            </div>
        `;

        html += `
            <div class="section">
                <div class="section-title">Y (Process Output Magnitude)</div>
                ${renderExpression(data.y_analysis)}
            </div>
        `;
    }

    content.innerHTML = html;

    // Render math
    renderMath();
}

// Display flow details in panel
function displayFlowDetails(data) {
    const panel = document.getElementById('details-panel');
    const title = document.getElementById('panel-title');
    const content = document.getElementById('panel-content');

    title.textContent = `Flow: ${data.material}`;

    let html = `
        <div class="section">
            <div class="section-title">Flow Information</div>
            <div class="info-grid">
                <div class="info-label">Material:</div>
                <div class="info-value">${data.material}</div>
                <div class="info-label">Type:</div>
                <div class="info-value">${data.flow_type}</div>
                <div class="info-label">From:</div>
                <div class="info-value">${data.source}</div>
                <div class="info-label">To:</div>
                <div class="info-value">${data.target}</div>
                <div class="info-label">Value:</div>
                <div class="info-value">${data.evaluated_value}</div>
            </div>
            <p style="margin-top: 1rem; color: #7f8c8d; font-size: 0.9rem;">${data.description}</p>
        </div>
    `;

    // Flow expression
    html += `
        <div class="section">
            <div class="section-title">Flow Expression</div>
            ${renderExpression(data.analysis)}
        </div>
    `;

    content.innerHTML = html;

    // Render math
    renderMath();
}

// Render expression analysis
function renderExpression(analysis) {
    let html = '';

    // Show all three evaluation modes if available
    if (analysis.evaluation_modes) {
        html += `
            <div class="expression-box">
                <div class="expression-label">Expression (Fully Symbolic)</div>
                <div class="expression-content">${escapeHtml(analysis.evaluation_modes.symbolic)}</div>
                <div class="latex-content">
                    $$${analysis.evaluation_modes.symbolic_latex}$$
                </div>
                <div style="font-size: 0.85rem; color: #7f8c8d; margin-top: 0.5rem;">
                    No substitution - shows model structure
                </div>
            </div>

            <div class="expression-box">
                <div class="expression-label">Expression (Recipe Evaluated)</div>
                <div class="expression-content">${escapeHtml(analysis.evaluation_modes.recipe_evaluated)}</div>
                <div class="latex-content">
                    $$${analysis.evaluation_modes.recipe_latex}$$
                </div>
                <div style="font-size: 0.85rem; color: #7f8c8d; margin-top: 0.5rem;">
                    Recipe coefficients (S, U) substituted
                </div>
                ${analysis.evaluation_modes.missing_coefficients && analysis.evaluation_modes.missing_coefficients.length > 0 ? `
                    <div style="background: #fff3cd; border: 1px solid #ffc107; border-radius: 4px; padding: 0.75rem; margin-top: 0.5rem;">
                        <div style="color: #856404; font-weight: 600; font-size: 0.9rem; margin-bottom: 0.25rem;">
                            ⚠️ Warning: Missing Recipe Coefficients
                        </div>
                        <div style="color: #856404; font-size: 0.85rem;">
                            The following coefficients are not defined in recipe_data:
                            <code style="background: #fff; padding: 0.2rem 0.4rem; border-radius: 3px; margin-left: 0.25rem;">
                                ${analysis.evaluation_modes.missing_coefficients.join(', ')}
                            </code>
                        </div>
                        <div style="color: #856404; font-size: 0.8rem; margin-top: 0.5rem; font-style: italic;">
                            This typically indicates a typo in the recipe_data dictionary or missing coefficient definitions.
                        </div>
                    </div>
                ` : ''}
            </div>

            <div class="expression-box">
                <div class="expression-label">Expression (Fully Evaluated)</div>
                <div class="expression-content">${escapeHtml(analysis.evaluation_modes.fully_evaluated)}</div>
                <div class="latex-content">
                    $$${analysis.evaluation_modes.fully_latex}$$
                </div>
                <div style="font-size: 0.85rem; color: #7f8c8d; margin-top: 0.5rem;">
                    All parameters and intermediates substituted
                </div>
            </div>
        `;
    } else {
        // Fallback to old format if evaluation_modes not available
        html += `
            <div class="expression-box">
                <div class="expression-label">Final Expression</div>
                <div class="expression-content">${escapeHtml(analysis.final_expression)}</div>
                <div class="latex-content">
                    $$${analysis.final_latex}$$
                </div>
            </div>
        `;
    }

    // History
    if (analysis.history && analysis.history.length > 0) {
        html += `
            <div style="margin-top: 1.5rem;">
                <div class="expression-label">Modeling History</div>
                <p style="font-size: 0.85rem; color: #7f8c8d; margin-bottom: 0.5rem;">
                    Steps that contributed to this expression:
                </p>
        `;
        for (const step of analysis.history) {
            html += `<div class="history-item">${escapeHtml(step)}</div>`;
        }
        html += `</div>`;
    }

    // Intermediates
    if (analysis.intermediates && analysis.intermediates.length > 0) {
        html += `
            <div style="margin-top: 1.5rem;">
                <div class="expression-label">Intermediate Variables</div>
                <p style="font-size: 0.85rem; color: #7f8c8d; margin-bottom: 0.5rem;">
                    Breakdown of intermediate calculations:
                </p>
        `;
        for (const inter of analysis.intermediates) {
            html += `
                <div class="intermediate-item">
                    <div class="intermediate-symbol">${escapeHtml(inter.symbol)}</div>
                    <div class="intermediate-description">${escapeHtml(inter.description)}</div>
                    <div class="intermediate-value">${escapeHtml(inter.value)}</div>
                    <div class="latex-content">
                        $$${inter.symbol} = ${inter.value_latex}$$
                    </div>
                </div>
            `;
        }
        html += `</div>`;
    }

    return html;
}

// Utility functions
function clearSelection() {
    selectedItem = null;
    cy.elements().unselect();

    // Show empty state in panel
    const content = document.getElementById('panel-content');
    const title = document.getElementById('panel-title');
    title.textContent = 'Details';
    content.innerHTML = `
        <div class="empty-state">
            <div class="empty-state-icon">📊</div>
            <p>Click on a process or flow to see details</p>
        </div>
    `;
}

function fitGraph() {
    cy.fit(null, 50);
}

function resetLayout() {
    cy.layout({
        name: 'dagre',
        rankDir: 'LR',
        nodeSep: 50,
        rankSep: 100,
        padding: 30
    }).run();
}

function showError(message) {
    const content = document.getElementById('panel-content');
    content.innerHTML = `<div class="error">Error: ${escapeHtml(message)}</div>`;
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function renderMath() {
    if (window.MathJax) {
        MathJax.typesetPromise().catch((err) => console.log('MathJax error:', err));
    }
}

// ============================================================================
// Time-Travel Debugger Functions
// ============================================================================

async function loadStepsData() {
    try {
        console.log('Loading steps data...');
        const response = await fetch('/api/steps');
        const data = await response.json();
        stepsData = data.steps;
        const lastStep = Math.max(0, data.steps.length - 1);

        console.log(`Loaded ${stepsData.length} steps`);

        // Initialize slider
        const slider = document.getElementById('step-slider');
        slider.max = lastStep;
        slider.value = lastStep;

        // Compute global edge min/max across all steps for consistent width scaling
        await computeGlobalEdgeMinMax();

        // ALWAYS load the last step first to establish a consistent layout
        // This ensures all nodes get positioned even if they have no edges in early steps
        await loadGraphDataAtStep(lastStep, true); // true = is initial load

        // Now set the current step to the last step
        currentStep = lastStep;
        updateStepInfo();
        updateButtonStates();

    } catch (error) {
        console.error('Error loading steps:', error);
        showError('Failed to load time-travel steps');
    }
}

async function computeGlobalEdgeMinMax() {
    // Fetch all steps and find global min/max edge values
    const allFlowValues = [];

    for (let step = 0; step < stepsData.length; step++) {
        try {
            const response = await fetch(`/api/graph/${step}`);
            const data = await response.json();

            for (const edge of data.edges) {
                const numericValue = edge.data.numeric_value;
                if (numericValue !== null && numericValue !== undefined && numericValue > 0) {
                    allFlowValues.push(numericValue);
                }
            }
        } catch (error) {
            console.error(`Error fetching graph for step ${step}:`, error);
        }
    }

    if (allFlowValues.length > 0) {
        globalEdgeMinMax = {
            min: Math.min(...allFlowValues),
            max: Math.max(...allFlowValues)
        };
        console.log('Global edge min/max:', globalEdgeMinMax);
    }
}

async function loadGraphDataAtStep(step, isInitialLoad = false) {
    try {
        console.log(`Loading graph data for step ${step}...`);
        const response = await fetch(`/api/graph/${step}`);
        graphData = await response.json();
        console.log('Graph data loaded:', graphData.nodes.length, 'nodes,', graphData.edges.length, 'edges');

        // Clear existing graph
        cy.elements().remove();

        // Add elements to graph
        cy.add(graphData.nodes);
        cy.add(graphData.edges);

        // Apply edge widths based on flow values
        applyEdgeWidths();

        console.log('Elements added to Cytoscape');

        // Handle layout and positions
        if (isInitialLoad) {
            // First time - run layout on complete graph and save positions
            const layout = cy.layout({
                name: 'dagre',
                rankDir: 'LR',
                nodeSep: 50,
                rankSep: 100,
                padding: 30
            });

            // Wait for layout to complete before saving positions
            await new Promise((resolve) => {
                layout.on('layoutstop', function() {
                    savedPositions = {};
                    cy.nodes().forEach(node => {
                        const pos = node.position();
                        // Deep copy the position to avoid reference issues
                        savedPositions[node.id()] = { x: pos.x, y: pos.y };
                    });
                    console.log('Node positions saved:', Object.keys(savedPositions).length, 'nodes');
                    resolve();
                });
                layout.run();
            });

            // Fit to screen after layout
            cy.fit(null, 50);
        } else {
            // Restore saved positions for all nodes
            if (savedPositions) {
                cy.nodes().forEach(node => {
                    const savedPos = savedPositions[node.id()];
                    if (savedPos) {
                        node.position({ x: savedPos.x, y: savedPos.y });
                    } else {
                        console.warn('No saved position for node:', node.id());
                    }
                });
                console.log('Node positions restored');
            }
        }

        console.log('Graph layout complete');

    } catch (error) {
        console.error('Error loading graph data:', error);
        showError('Failed to load graph data');
    }
}

function applyEdgeWidths() {
    // Use global min/max for consistent edge widths across all steps
    if (!globalEdgeMinMax) {
        // Fallback: use default width if global min/max not computed
        cy.edges().forEach(edge => {
            edge.style('width', 3);
        });
        return;
    }

    const minValue = globalEdgeMinMax.min;
    const maxValue = globalEdgeMinMax.max;
    const range = maxValue - minValue;

    // Set edge widths based on normalized flow values using global min/max
    cy.edges().forEach(edge => {
        const numericValue = edge.data('numeric_value');
        if (numericValue !== null && numericValue !== undefined && numericValue > 0) {
            // Normalize to 1-10 pixel range using global scale
            let width;
            if (range > 0) {
                width = 1 + ((numericValue - minValue) / range) * 9;
            } else {
                width = 5; // Default if all values are the same
            }
            edge.style('width', width);
        } else {
            edge.style('width', 2); // Default for symbolic/zero values
        }
    });
}

function updateStepInfo() {
    const stepText = document.getElementById('step-text');
    if (currentStep >= 0 && currentStep < stepsData.length) {
        const stepData = stepsData[currentStep];
        stepText.textContent = `Step ${currentStep + 1}/${stepsData.length}: ${stepData.label}`;
    } else {
        stepText.textContent = 'No steps available';
    }
}

function updateButtonStates() {
    const firstBtn = document.getElementById('first-step-btn');
    const prevBtn = document.getElementById('prev-step-btn');
    const nextBtn = document.getElementById('next-step-btn');
    const lastBtn = document.getElementById('last-step-btn');

    const isFirstStep = currentStep <= 0;
    const isLastStep = currentStep >= stepsData.length - 1;

    firstBtn.disabled = isFirstStep;
    prevBtn.disabled = isFirstStep;
    nextBtn.disabled = isLastStep;
    lastBtn.disabled = isLastStep;
}

function goToStep(step) {
    if (step < 0 || step >= stepsData.length) return;

    currentStep = step;

    // Update slider
    document.getElementById('step-slider').value = currentStep;

    // Update UI
    updateStepInfo();
    updateButtonStates();

    // Load graph for this step
    loadGraphDataAtStep(currentStep);

    // Refresh the selected item's details at this step
    if (selectedItem) {
        refreshSelectedItem();
    }
}

function refreshSelectedItem() {
    if (!selectedItem) return;

    if (selectedItem.type === 'process') {
        loadProcessDetails(selectedItem.id);
    } else if (selectedItem.type === 'flow') {
        loadFlowDetails(selectedItem.source, selectedItem.target, selectedItem.material);
    }
}

function previousStep() {
    if (currentStep > 0) {
        goToStep(currentStep - 1);
    }
}

function nextStep() {
    if (currentStep < stepsData.length - 1) {
        goToStep(currentStep + 1);
    }
}

function goToFirstStep() {
    goToStep(0);
}

function goToLastStep() {
    goToStep(stepsData.length - 1);
}

function sliderChanged(value) {
    goToStep(parseInt(value));
}

function togglePlayPause() {
    const btn = document.getElementById('play-pause-btn');

    if (isPlaying) {
        // Pause
        isPlaying = false;
        clearInterval(playInterval);
        playInterval = null;
        btn.textContent = '▶ Play';
    } else {
        // Play
        isPlaying = true;
        btn.textContent = '⏸ Pause';

        // Auto-advance every 1.5 seconds
        playInterval = setInterval(() => {
            if (currentStep < stepsData.length - 1) {
                nextStep();
            } else {
                // Reached the end, stop playing
                togglePlayPause();
            }
        }, 1500);
    }
}

async function loadProcessDetails(processId) {
    try {
        // Track what's selected
        selectedItem = { type: 'process', id: processId };

        console.log(`Fetching process details for: ${processId} at step ${currentStep}`);
        const response = await fetch(`/api/process/${currentStep}/${encodeURIComponent(processId)}`);
        const data = await response.json();
        console.log('Process details received:', data);

        if (data.error) {
            console.error('API returned error:', data.error);
            showError(data.error);
            return;
        }

        displayProcessDetails(data);

    } catch (error) {
        console.error('Error loading process details:', error);
        showError('Failed to load process details');
    }
}

async function loadFlowDetails(source, target, material) {
    try {
        // Track what's selected
        selectedItem = { type: 'flow', source, target, material };

        const response = await fetch(`/api/flow/${currentStep}/${encodeURIComponent(source)}/${encodeURIComponent(target)}/${encodeURIComponent(material)}`);
        const data = await response.json();

        if (data.error) {
            showError(data.error);
            return;
        }

        displayFlowDetails(data);

    } catch (error) {
        console.error('Error loading flow details:', error);
        showError('Failed to load flow details');
    }
}
