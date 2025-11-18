// FlowProg Visualization Application

let cy; // Cytoscape instance
let graphData = null;

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
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
        loadProcessDetails(node.data('id'));
    });

    cy.on('tap', 'edge.flow', function(evt) {
        const edge = evt.target;
        const source = edge.data('source');
        const target = edge.data('target');
        const material = edge.data('material');
        loadFlowDetails(source, target, material);
    });

    // Double-click to deselect
    cy.on('tap', function(evt) {
        if (evt.target === cy) {
            closePanel();
        }
    });
}

// Load graph data from API
async function loadGraphData() {
    try {
        const response = await fetch('/api/graph');
        graphData = await response.json();

        // Add elements to graph
        cy.add(graphData.nodes);
        cy.add(graphData.edges);

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

    } catch (error) {
        console.error('Error loading graph data:', error);
        showError('Failed to load graph data');
    }
}

// Load process details
async function loadProcessDetails(processId) {
    try {
        const response = await fetch(`/api/process/${encodeURIComponent(processId)}`);
        const data = await response.json();

        if (data.error) {
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
    const panel = document.getElementById('details-panel');
    const title = document.getElementById('panel-title');
    const content = document.getElementById('panel-content');

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
    panel.classList.remove('collapsed');

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

    // Show panel
    panel.classList.remove('collapsed');

    // Render math
    renderMath();
}

// Render expression analysis
function renderExpression(analysis) {
    let html = `
        <div class="expression-box">
            <div class="expression-label">Final Expression</div>
            <div class="expression-content">${escapeHtml(analysis.final_expression)}</div>
            <div class="latex-content">
                $$${analysis.final_latex}$$
            </div>
        </div>
    `;

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
