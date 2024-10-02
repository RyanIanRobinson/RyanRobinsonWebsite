document.getElementById("runModelButton").addEventListener("click", async () => {
    try {
        const response = await fetch('/runModel');
        
        if (response.ok) {
            const data = await response.json();
            // Display the results from Flask
            document.getElementById("results").innerHTML = `
                <h3>Current Close Price: ${data.current_close_price}</h3>
                <h4>Future Predictions:</h4>
                <table>
                    <tr>
                        <th>Day</th><th>Predicted Close Price</th><th>CI Lower 95%</th><th>CI Upper 95%</th>
                        <th>CI Lower 90%</th><th>CI Upper 90%</th><th>CI Lower 80%</th><th>CI Upper 80%</th>
                    </tr>
                    ${data.predictions.map(prediction => `
                        <tr>
                            <td>${prediction.Day}</td>
                            <td>${prediction.Predicted_Close_Price}</td>
                            <td>${prediction['CI_Lower_95%']}</td>
                            <td>${prediction['CI_Upper_95%']}</td>
                            <td>${prediction['CI_Lower_90%']}</td>
                            <td>${prediction['CI_Upper_90%']}</td>
                            <td>${prediction['CI_Lower_80%']}</td>
                            <td>${prediction['CI_Upper_80%']}</td>
                        </tr>`).join('')}
                </table>
                <h4>Trading Decision: ${data.decision}</h4>
            `;
        } else {
            document.getElementById("results").innerText = "Error running model. Please try again.";
        }
    } catch (error) {
        console.error("Error:", error);
        document.getElementById("results").innerText = "Error running model. Please try again.";
    }
});
