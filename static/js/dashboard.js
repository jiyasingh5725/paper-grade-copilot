/* ============================================================
   PAPER GRADE TRANSITION COPILOT
   Frontend API Controller
============================================================ */

const API_BASE = "";

let currentProcess = null;
let latestPrediction = null;
let latestRecommendation = null;
let trajectoryChart = null;


/* ============================================================
   HELPER
============================================================ */

function getElement(id) {
    return document.getElementById(id);
}


function formatNumber(value, decimals = 3) {

    if (
        value === null ||
        value === undefined ||
        value === "" ||
        isNaN(value)
    ) {
        return "--";
    }

    return Number(value).toFixed(decimals);
}


/* ============================================================
   SYSTEM HEALTH
============================================================ */

async function loadHealth() {

    try {

        const response = await fetch(
            `${API_BASE}/api/health`
        );

        const data = await response.json();

        const statusText =
            getElement("systemStatus");

        const statusDot =
            getElement("statusDot");


        if (
            data.status === "running" &&
            data.models_loaded &&
            data.dataset_loaded
        ) {

            statusText.textContent =
                "SYSTEM ONLINE";

            statusDot.style.background =
                "#198754";

        } else {

            statusText.textContent =
                "SYSTEM WARNING";

            statusDot.style.background =
                "#f0a500";
        }

    } catch (error) {

        console.error(
            "Health check failed:",
            error
        );

        getElement("systemStatus").textContent =
            "API OFFLINE";

        getElement("statusDot").style.background =
            "#dc3545";
    }
}


/* ============================================================
   CURRENT PROCESS
============================================================ */

async function loadCurrentProcess() {

    try {

        const response = await fetch(
            `${API_BASE}/api/current-process`
        );

        const data = await response.json();


        if (!data.success) {

            throw new Error(
                data.message ||
                "Could not load current process."
            );
        }


        currentProcess = data;


        /* -----------------------------------------------------
           Transition
        ----------------------------------------------------- */

        getElement("transitionId")
            .textContent =
            data.transition_id || "--";


        getElement("gradeFrom")
            .textContent =
            data.grade_from || "--";


        getElement("gradeTo")
            .textContent =
            data.grade_to || "--";


        /* -----------------------------------------------------
           Process inputs
        ----------------------------------------------------- */

        getElement("machineSpeed")
            .value =
            formatNumber(
                data.machine_speed
            );


        getElement("stockFlow")
            .value =
            formatNumber(
                data.stock_flow
            );


        getElement("steamPressure")
            .value =
            formatNumber(
                data.steam_pressure
            );


        /* -----------------------------------------------------
           Current values
        ----------------------------------------------------- */

        getElement("currentBasisWeight")
            .textContent =
            formatNumber(
                data.basis_weight
            );


        getElement("targetSetpoint")
            .textContent =
            formatNumber(
                data.basis_weight_setpoint
            );


        getElement("metricCurrent")
            .textContent =
            formatNumber(
                data.basis_weight
            );


        getElement("metricTarget")
            .textContent =
            formatNumber(
                data.basis_weight_setpoint
            );


    } catch (error) {

        console.error(
            "Current process error:",
            error
        );

        getElement("systemStatus")
            .textContent =
            "DATA ERROR";
    }
}


/* ============================================================
   INPUT VALIDATION
============================================================ */

function validateInputs() {

    const machineSpeed =
        Number(
            getElement("machineSpeed").value
        );

    const stockFlow =
        Number(
            getElement("stockFlow").value
        );

    const steamPressure =
        Number(
            getElement("steamPressure").value
        );


    if (
        isNaN(machineSpeed) ||
        machineSpeed < 700 ||
        machineSpeed > 900
    ) {

        alert(
            "Machine Speed must be between 700 and 900 m/min."
        );

        return false;
    }


    if (
        isNaN(stockFlow) ||
        stockFlow < 50 ||
        stockFlow > 80
    ) {

        alert(
            "Stock Flow must be between 50 and 80%."
        );

        return false;
    }


    if (
        isNaN(steamPressure) ||
        steamPressure < 4 ||
        steamPressure > 6
    ) {

        alert(
            "Steam Pressure must be between 4 and 6 bar."
        );

        return false;
    }


    return true;
}


/* ============================================================
   PREDICTION
============================================================ */

async function runPrediction() {

    if (!validateInputs()) {
        return;
    }


    const button =
        getElement("analyzeButton");


    button.disabled = true;

    button.innerHTML =
        "<span>ANALYZING...</span><span>⏳</span>";


    try {

        const payload = {

            machine_speed:
                Number(
                    getElement("machineSpeed").value
                ),

            stock_flow:
                Number(
                    getElement("stockFlow").value
                ),

            steam_pressure:
                Number(
                    getElement("steamPressure").value
                )
        };


        console.log(
            "Sending prediction request:",
            payload
        );


        const response = await fetch(
            `${API_BASE}/api/predict`,
            {
                method: "POST",

                headers: {
                    "Content-Type":
                        "application/json"
                },

                body:
                    JSON.stringify(payload)
            }
        );


        const data =
            await response.json();


        console.log(
            "Prediction result:",
            data
        );


        if (!data.success) {

            throw new Error(
                data.message ||
                data.error ||
                "Prediction failed."
            );
        }


        latestPrediction = data;


        displayPrediction(
            data
        );


        createTrajectoryChart(
            data
        );


        updateExplanation(
            data
        );


    } catch (error) {

        console.error(
            "Prediction error:",
            error
        );

        alert(
            `Prediction failed: ${error.message}`
        );

    } finally {

        button.disabled = false;

        button.innerHTML =
            "<span>ANALYZE TRANSITION</span><span>→</span>";
    }
}


/* ============================================================
   DISPLAY PREDICTION
============================================================ */

function displayPrediction(data) {

    getElement("metricPredicted")
        .textContent =
        formatNumber(
            data.predicted_basis_weight
        );


    getElement("metricDeviation")
        .textContent =
        `${formatNumber(
            data.predicted_deviation_pct
        )}%`;


    const statusElement =
        getElement("predictionStatus");

    const statusText =
        getElement("predictionStatusText");

    const description =
        getElement("predictionDescription");

    const riskIcon =
        getElement("riskIcon");


    if (Number(data.off_spec) === 1) {

        /* -----------------------------------------------------
           RED / HIGH RISK
        ----------------------------------------------------- */

        statusText.textContent =
            "🔴 OFF-SPEC / HIGH RISK";

        riskIcon.textContent =
            "🔴";

        description.textContent =
            `Predicted Basis Weight is ${formatNumber(
                data.predicted_basis_weight
            )} GSM, which is ${formatNumber(
                data.predicted_deviation_pct,
                2
            )}% away from the target.`;

        statusElement.style.background =
            "#fff0f0";

        statusElement.style.borderColor =
            "#e6b8b8";

    } else {

        /* -----------------------------------------------------
           GREEN / SAFE
        ----------------------------------------------------- */

        statusText.textContent =
            "🟢 SAFE PREDICTION";

        riskIcon.textContent =
            "🟢";

        description.textContent =
            `Predicted Basis Weight is within the 2.5% specification limit.`;

        statusElement.style.background =
            "#edf8f2";

        statusElement.style.borderColor =
            "#b9dfc7";
    }
}


/* ============================================================
   TRAJECTORY CHART
============================================================ */

function createTrajectoryChart(data) {

    const canvas =
        getElement("trajectoryChart");


    if (!canvas) {
        return;
    }


    const current =
        Number(
            currentProcess?.basis_weight || 0
        );


    const target =
        Number(
            data.setpoint || 0
        );


    const predicted =
        Number(
            data.predicted_basis_weight || 0
        );


    /*
     * The current backend returns only the current state
     * and predicted future value.
     *
     * Therefore these intermediate points are a visual
     * interpolation, not measured future sensor data.
     */

    const labels = [
        "Current",
        "1 min",
        "2 min",
        "3 min",
        "4 min",
        "5 min"
    ];


    const trajectory = [
        current,
        current + (predicted - current) * 0.20,
        current + (predicted - current) * 0.40,
        current + (predicted - current) * 0.60,
        current + (predicted - current) * 0.80,
        predicted
    ];


    if (trajectoryChart) {

        trajectoryChart.destroy();

    }


    trajectoryChart =
        new Chart(
            canvas,
            {

                type: "line",

                data: {

                    labels: labels,

                    datasets: [

                        {
                            label:
                                "Predicted Basis Weight",

                            data:
                                trajectory,

                            borderWidth: 3,

                            tension: 0.35,

                            pointRadius: 4
                        },

                        {
                            label:
                                "Target Setpoint",

                            data:
                                labels.map(
                                    () => target
                                ),

                            borderWidth: 2,

                            borderDash: [
                                6,
                                6
                            ],

                            pointRadius: 0
                        }

                    ]
                },


                options: {

                    responsive: true,

                    maintainAspectRatio: false,

                    plugins: {

                        legend: {
                            display: true
                        }

                    },

                    scales: {

                        y: {

                            title: {

                                display: true,

                                text:
                                    "Basis Weight (GSM)"
                            }

                        },

                        x: {

                            title: {

                                display: true,

                                text:
                                    "Transition Time"
                            }

                        }

                    }

                }

            }
        );
}


/* ============================================================
   RECOMMENDATIONS
============================================================ */

async function loadRecommendations() {

    try {

        const response = await fetch(
            `${API_BASE}/api/recommendations`
        );

        const data =
            await response.json();


        if (!data.success) {

            throw new Error(
                data.message ||
                "Recommendations unavailable."
            );
        }


        if (
            !data.recommendations ||
            data.recommendations.length === 0
        ) {

            getElement("recommendationStatus")
                .textContent =
                "NO RECOMMENDATION FOUND";

            return;
        }


        /*
         * Stage 5 has already ranked the recommendation
         * candidates.
         */

        const best =
            data.recommendations[0];


        latestRecommendation =
            best;


        displayRecommendation(
            best
        );


    } catch (error) {

        console.error(
            "Recommendation error:",
            error
        );

        getElement("recommendationStatus")
            .textContent =
            "RECOMMENDATION DATA UNAVAILABLE";
    }
}


/* ============================================================
   DISPLAY RECOMMENDATION
============================================================ */

function displayRecommendation(
    recommendation
) {

    getElement("recommendedSpeed")
        .textContent =
        formatNumber(
            recommendation.machine_speed
        );


    getElement("recommendedStock")
        .textContent =
        formatNumber(
            recommendation.stock_flow
        );


    getElement("recommendedSteam")
        .textContent =
        formatNumber(
            recommendation.steam_pressure
        );


    getElement("recommendedBW")
        .textContent =
        formatNumber(
            recommendation.predicted_basis_weight
        );


    getElement("recommendedDeviation")
        .textContent =
        `${formatNumber(
            recommendation.predicted_deviation_pct
        )}%`;


    const status =
        Number(
            recommendation.off_spec
        );


    const banner =
        getElement("recommendationBanner");


    const title =
        getElement("recommendationStatus");


    if (status === 0) {

        title.textContent =
            "🟢 SAFE RECOMMENDATION FOUND";

        banner.style.background =
            "#edf8f2";

        banner.style.borderColor =
            "#b9dfc7";


        getElement("operatorReviewText")
            .textContent =
            "The recommended operating point is predicted to remain within the 2.5% Basis Weight specification limit. Operator review is required before applying the process change.";

    } else {

        title.textContent =
            "🔴 NO SAFE RECOMMENDATION FOUND";

        banner.style.background =
            "#fff0f0";

        banner.style.borderColor =
            "#e6b8b8";


        getElement("operatorReviewText")
            .textContent =
            "No tested operating combination was predicted to satisfy the 2.5% specification limit. Do not automatically apply an unsafe operating condition.";
    }
}


/* ============================================================
   ACCEPT RECOMMENDATION
============================================================ */

function acceptRecommendation() {

    if (!latestRecommendation) {

        alert(
            "No recommendation is available to accept."
        );

        return;
    }


    const review =
        getElement("operatorReview");


    review.classList.remove(
        "rejected"
    );

    review.classList.add(
        "accepted"
    );


    getElement("operatorReviewText")
        .textContent =
        "Recommendation accepted for operator review. The system does not automatically change machine parameters.";
}


/* ============================================================
   REJECT RECOMMENDATION
============================================================ */

function rejectRecommendation() {

    if (!latestRecommendation) {

        alert(
            "No recommendation is available to reject."
        );

        return;
    }


    const review =
        getElement("operatorReview");


    review.classList.remove(
        "accepted"
    );

    review.classList.add(
        "rejected"
    );


    getElement("operatorReviewText")
        .textContent =
        "Recommendation rejected by the operator. No process parameters have been changed.";
}


/* ============================================================
   HISTORICAL EVIDENCE
============================================================ */

async function loadEvidence() {

    try {

        const response = await fetch(
            `${API_BASE}/api/evidence`
        );


        const data = await response.json();


        console.log(
            "Stage 6 evidence response:",
            data
        );


        if (!response.ok || !data.success) {

            throw new Error(
                data.message ||
                "Evidence unavailable."
            );
        }


        const evidence =
            data.evidence || {};


        /* -----------------------------------------------------
           HISTORICAL EVIDENCE OBJECT
        ----------------------------------------------------- */

        const historical =
            evidence.historical_evidence || {};


        /* -----------------------------------------------------
           READ EXACT BACKEND FIELD NAMES
        ----------------------------------------------------- */

        const records =
            historical.historical_transition_records;


        const matchingObservations =
            historical.matching_grade_transition_observations;


        const success =
            historical.success_rate_pct;


        const failure =
            historical.failure_rate_pct;


        const offSpec =
            historical.off_spec_rate_pct;


        /* -----------------------------------------------------
           STABILIZATION OBJECT
        ----------------------------------------------------- */

        const stabilization =
            evidence.stabilization || {};


        const stabilizationMinutes =
            stabilization.estimated_stabilization_time_min;


        const medianStabilization =
            historical.median_stabilization_time_min;


        const minStabilization =
            historical.minimum_stabilization_time_min;


        const maxStabilization =
            historical.maximum_stabilization_time_min;


        /* -----------------------------------------------------
           DISPLAY HISTORICAL RECORDS
        ----------------------------------------------------- */

        getElement("evidenceRecords")
            .textContent =
            records !== undefined
                ? records
                : "--";


        /* -----------------------------------------------------
           DISPLAY SUCCESS RATE
        ----------------------------------------------------- */

        getElement("successRate")
            .textContent =
            success !== undefined
                ? `${formatNumber(success, 2)}%`
                : "--";


        /* -----------------------------------------------------
           DISPLAY FAILURE RATE
        ----------------------------------------------------- */

        getElement("failureRate")
            .textContent =
            failure !== undefined
                ? `${formatNumber(failure, 2)}%`
                : "--";


        /* -----------------------------------------------------
           DISPLAY STABILIZATION TIME
        ----------------------------------------------------- */

        getElement("stabilizationTime")
            .textContent =
            stabilizationMinutes !== undefined
                ? `${formatNumber(
                    stabilizationMinutes,
                    1
                )} min`
                : "--";


        /* -----------------------------------------------------
           UPDATE EVIDENCE MESSAGE
        ----------------------------------------------------- */

        getElement("evidenceMessage")
            .textContent =
            `Historical A → D transition analysis contains ${records} historical transition records and ${matchingObservations} matching grade-transition observations. The historical success rate is ${formatNumber(success, 2)}%, with an average stabilization time of ${formatNumber(stabilizationMinutes, 1)} minutes. Evidence strength: ${stabilization.historical_evidence_strength || "N/A"}.`;


        /* -----------------------------------------------------
           OPTIONAL CONSOLE INFORMATION
           Useful for debugging / demo
        ----------------------------------------------------- */

        console.log(
            "Historical transition records:",
            records
        );


        console.log(
            "Matching observations:",
            matchingObservations
        );


        console.log(
            "Success rate:",
            success
        );


        console.log(
            "Failure rate:",
            failure
        );


        console.log(
            "Off-spec rate:",
            offSpec
        );


        console.log(
            "Average stabilization:",
            stabilizationMinutes
        );


        console.log(
            "Median stabilization:",
            medianStabilization
        );


        console.log(
            "Minimum stabilization:",
            minStabilization
        );


        console.log(
            "Maximum stabilization:",
            maxStabilization
        );


    } catch (error) {

        console.error(
            "Evidence error:",
            error
        );


        getElement("evidenceRecords")
            .textContent =
            "--";


        getElement("successRate")
            .textContent =
            "--";


        getElement("failureRate")
            .textContent =
            "--";


        getElement("stabilizationTime")
            .textContent =
            "--";


        getElement("evidenceMessage")
            .textContent =
            `Historical evidence could not be loaded: ${error.message}`;
    }
}


/* ============================================================
   EXPLAINABILITY
============================================================ */

function updateExplanation(
    prediction = null
) {

    const container =
        getElement(
            "explanationContent"
        );


    if (!currentProcess) {
        return;
    }


    const currentBW =
        Number(
            currentProcess.basis_weight
        );


    const target =
        Number(
            currentProcess.basis_weight_setpoint
        );


    const predicted =
        prediction
            ? Number(
                prediction.predicted_basis_weight
            )
            : null;


    const deviation =
        target !== 0
            ? Math.abs(
                currentBW - target
            ) / target * 100
            : 0;


    const gradeFrom =
        currentProcess.grade_from;


    const gradeTo =
        currentProcess.grade_to;


    const explanations = [];


    /* ---------------------------------------------------------
       Explanation 1
    --------------------------------------------------------- */

    if (currentBW < target) {

        explanations.push(
            `Current Basis Weight is ${formatNumber(
                Math.abs(currentBW - target),
                3
            )} GSM below the ${formatNumber(
                target
            )} GSM target.`
        );

    } else if (currentBW > target) {

        explanations.push(
            `Current Basis Weight is ${formatNumber(
                Math.abs(currentBW - target),
                3
            )} GSM above the ${formatNumber(
                target
            )} GSM target.`
        );

    } else {

        explanations.push(
            "Current Basis Weight is aligned with the target setpoint."
        );
    }


    /* ---------------------------------------------------------
       Explanation 2
    --------------------------------------------------------- */

    explanations.push(
        `Current deviation is ${formatNumber(
            deviation,
            2
        )}%. The prototype uses a 2.5% specification threshold.`
    );


    /* ---------------------------------------------------------
       Explanation 3
    --------------------------------------------------------- */

    if (predicted !== null) {

        const predictedDeviation =
            target !== 0
                ? Math.abs(
                    predicted - target
                ) / target * 100
                : 0;


        if (predictedDeviation > 2.5) {

            explanations.push(
                `The AI predicts ${formatNumber(
                    predicted
                )} GSM after the transition, resulting in ${formatNumber(
                    predictedDeviation,
                    2
                )}% deviation and therefore an OFF-SPEC risk.`
            );

        } else {

            explanations.push(
                `The AI predicts ${formatNumber(
                    predicted
                )} GSM after the transition, remaining within the 2.5% specification limit.`
            );
        }

    }


    /* ---------------------------------------------------------
       Explanation 4
    --------------------------------------------------------- */

    explanations.push(
        `The system is evaluating the ${gradeFrom} → ${gradeTo} grade transition using historical evidence from Stage 6.`
    );


    container.innerHTML = "";


    explanations.forEach(
        (text, index) => {

            const item =
                document.createElement(
                    "div"
                );


            item.className =
                "explanation-item";


            item.innerHTML = `

                <span class="explanation-number">
                    ${String(index + 1).padStart(2, "0")}
                </span>

                <p>
                    ${text}
                </p>

            `;


            container.appendChild(
                item
            );
        }
    );
}


/* ============================================================
   DASHBOARD INITIALIZATION
============================================================ */

async function initializeDashboard() {

    await loadHealth();

    await loadCurrentProcess();

    await loadRecommendations();

    await loadEvidence();

    updateExplanation();
}


/* ============================================================
   EVENT LISTENERS
============================================================ */

document.addEventListener(
    "DOMContentLoaded",
    () => {

        initializeDashboard();


        getElement("analyzeButton")
            .addEventListener(
                "click",
                runPrediction
            );


        getElement("acceptRecommendation")
            .addEventListener(
                "click",
                acceptRecommendation
            );


        getElement("rejectRecommendation")
            .addEventListener(
                "click",
                rejectRecommendation
            );

    }
);