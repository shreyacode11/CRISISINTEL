(function () {
    const latEl = document.getElementById("lat");
    const lonEl = document.getElementById("lon");
    const cityEl = document.getElementById("cityInput");
    const status = document.getElementById("locStatus");
    const form = document.getElementById("reqForm");

    let locationResolved = false;

    if (!navigator.geolocation) {
        status.innerHTML = '<i class="fa fa-triangle-exclamation text-warning me-1"></i>Geolocation not supported. Please enter city manually.';
        locationResolved = true;
        return;
    }

    navigator.geolocation.getCurrentPosition(
        async (pos) => {
            latEl.value = pos.coords.latitude;
            lonEl.value = pos.coords.longitude;
            status.innerHTML = `<i class="fa fa-circle-check text-success me-1"></i>
                Location captured (${pos.coords.latitude.toFixed(4)}, ${pos.coords.longitude.toFixed(4)})`;

            // Reverse geocode to auto-fill city
            try {
                const url = `https://nominatim.openstreetmap.org/reverse?format=json&lat=${pos.coords.latitude}&lon=${pos.coords.longitude}`;
                const res = await fetch(url, { headers: { "Accept": "application/json" } });
                const data = await res.json();
                const city =
                    data.address.city ||
                    data.address.town ||
                    data.address.village ||
                    data.address.county ||
                    data.address.state || "";
                if (city && !cityEl.value) cityEl.value = city;
            } catch (e) { /* silent */ }

            locationResolved = true;
        },
        (err) => {
            status.innerHTML = '<i class="fa fa-triangle-exclamation text-warning me-1"></i>' +
                'Location permission denied. Please type your city manually.';
            locationResolved = true;
        },
        { enableHighAccuracy: true, timeout: 8000, maximumAge: 60000 }
    );

    // Safety net: if user submits before location resolves, allow but warn
    if (form) {
        form.addEventListener("submit", (e) => {
            if (!locationResolved) {
                status.innerHTML = '<i class="fa fa-spinner fa-spin me-1"></i>Waiting for location…';
                e.preventDefault();
                setTimeout(() => {
                    form.requestSubmit();
                }, 1500);
            }
        });
    }
})();