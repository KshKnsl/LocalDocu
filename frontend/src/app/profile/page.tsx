import { UserProfile } from "@clerk/nextjs";

const ProfilePage = () => (
  <div className="flex items-center justify-center h-full">
    <UserProfile />
  </div>
);

export default ProfilePage;
