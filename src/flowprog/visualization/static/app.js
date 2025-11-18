// FlowProg Visualization Application

let cy; // Cytoscape instance
let graphData = null;

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    console.log('DOM Content Loaded');

    // Check initial panel state
    const panel = document.getElementById('details-panel');
    if (panel) {
        console.log('Initial panel state:');
        console.log('  className:', panel.className);
        console.log('  classList:', panel.classList);
        console.log('  classList.contains("collapsed"):', panel.classList.contains('collapsed'));
        const style = window.getComputedStyle(panel);
        console.log('  computed width:', style.width);
        console.log('  computed overflow:', style.overflow);
    } else {
        console.error('Panel element not found on page load!');
    }

    initializeCytoscape();
    loadGraphData();
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

    // Double-click to deselect
    cy.on('tap', function(evt) {
        if (evt.target === cy) {
            console.log('Background clicked, closing panel');
            closePanel();
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

    // X expression (Process Input Magnitude)
    html += `
        <div class="section">
            <div class="section-title">X (Process Input Magnitude)</div>
            ${renderExpression(data.x_analysis)}
        </div>
    `;

    // Y expression (Process Output Magnitude)
    html += `
        <div class="section">
            <div class="section-title">Y (Process Output Magnitude)</div>
            ${renderExpression(data.y_analysis)}
        </div>
    `;

    content.innerHTML = html;

    // Show panel
    console.log('Removing collapsed class from panel...');
    console.log('Panel classes before:', panel.className);
    console.log('Panel classList contains collapsed?', panel.classList.contains('collapsed'));

    // Get computed styles
    const computedStyle = window.getComputedStyle(panel);
    console.log('Panel computed width before:', computedStyle.width);
    console.log('Panel computed overflow before:', computedStyle.overflow);
    console.log('Panel computed display:', computedStyle.display);

    panel.classList.remove('collapsed');

    console.log('Panel classes after:', panel.className);
    console.log('Panel classList contains collapsed after?', panel.classList.contains('collapsed'));

    // Force the width to be set (workaround for flex layout issues)
    panel.style.width = '500px';

    // Check computed styles after
    const computedStyleAfter = window.getComputedStyle(panel);
    console.log('Panel computed width after:', computedStyleAfter.width);
    console.log('Panel offsetWidth:', panel.offsetWidth);
    console.log('Panel scrollHeight:', panel.scrollHeight);

    // Render math
    renderMath();
    console.log('Process details displayed successfully');
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

    // Show panel
    panel.classList.remove('collapsed');
    panel.style.width = '500px';

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
                <div style="font-size: 0.85rem; color: #7f8c8d; margin-top: 0.5rem;">
                    No substitution - shows model structure
                </div>
            </div>

            <div class="expression-box">
                <div class="expression-label">Expression (Recipe Evaluated)</div>
                <div class="expression-content">${escapeHtml(analysis.evaluation_modes.recipe_evaluated)}</div>
                <div style="font-size: 0.85rem; color: #7f8c8d; margin-top: 0.5rem;">
                    Recipe coefficients (S, U) substituted
                </div>
            </div>

            <div class="expression-box">
                <div class="expression-label">Expression (Fully Evaluated)</div>
                <div class="expression-content">${escapeHtml(analysis.evaluation_modes.fully_evaluated)}</div>
                <div style="font-size: 0.85rem; color: #7f8c8d; margin-top: 0.5rem;">
                    All parameters substituted
                </div>
                <div class="latex-content">
                    $$${analysis.final_latex}$$
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
function closePanel() {
    const panel = document.getElementById('details-panel');
    panel.classList.add('collapsed');
    panel.style.width = '0';
    cy.elements().unselect();
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
    document.getElementById('details-panel').classList.remove('collapsed');
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
