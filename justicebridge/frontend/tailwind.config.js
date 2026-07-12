/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        sev: {
          red: "#c0392b",
          amber: "#e67e22",
          green: "#27ae60",
        },
      },
    },
  },
  plugins: [],
};
