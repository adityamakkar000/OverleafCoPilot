import Image from "next/image";

export default function Home() {
  return (
    <div className="flex min-h-screen">
      <div className="w-1/2 p-8 border-r">
      <form className="flex flex-col items-center justify-center gap-6">
        <textarea
        placeholder="Enter LaTeX content..."
        className="w-full h-64 p-6 text-lg border rounded-lg resize-none focus:outline-none focus:ring-2 focus:ring-blue-500 text-black"
        />
        <button
        type="submit"
        className="px-10 py-4 text-xl font-semibold text-white bg-blue-500 rounded-lg hover:bg-blue-600 transition-colors"
        >
        Convert LaTeX
        </button>
      </form>
      </div>

      <div className="w-1/2 p-8">
      <div className="h-full border rounded-lg p-6">
        <p className="text-gray-500">Generated output will appear here...</p>
      </div>
      </div>
    </div>

  );
}
