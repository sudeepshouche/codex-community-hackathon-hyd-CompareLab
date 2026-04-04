/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './src/pages/**/*.{js,ts,jsx,tsx,mdx}',
    './src/components/**/*.{js,ts,jsx,tsx,mdx}',
    './src/app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        background: 'hsl(var(--background))',
        foreground: 'hsl(var(--foreground))',
        muted: 'hsl(var(--muted))',
        border: 'hsl(var(--border))',
        surface: 'hsl(var(--surface))',
        'surface-2': 'hsl(var(--surface-2))',
        'tenant-primary': 'hsl(var(--tenant-primary))',
      },
      fontFamily: {
        sans: ['Inter', 'Geist Sans', 'Segoe UI', 'system-ui', 'sans-serif'],
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'skeleton': 'skeleton 2s linear infinite',
      },
      keyframes: {
        skeleton: {
          '0%': { backgroundColor: 'hsl(var(--muted) / 0.1)' },
          '50%': { backgroundColor: 'hsl(var(--muted) / 0.3)' },
          '100%': { backgroundColor: 'hsl(var(--muted) / 0.1)' },
        },
      },
    },
  },
  plugins: [],
}
